"""Local task queue for tracking Ray futures"""

import ray
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable


@dataclass
class Task:
    """Unified task with status-based state management"""
    task_id: str
    task_type: str
    config: Dict[str, Any]
    status: str = "pending"  # pending -> running -> completed/failed
    submit_callback: Optional[Callable] = None  # For pending tasks
    save_result_callback: Optional[Callable] = None  # For saving results when completed
    futures: Optional[List[Any]] = None  # For running tasks
    create_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0  # Number of retry attempts
    max_retries: int = 3  # Maximum retry attempts


class RayTaskQueue:
    """Local task queue for tracking Ray futures"""
    
    def __init__(self, ray_cluster_manager=None):
        self.tasks: Dict[str, Task] = {}  # All tasks by task_id
        self.task_order: List[str] = []  # Task IDs in submission order (FIFO)
        self.ray_cluster_manager = ray_cluster_manager  # For cleaning up placement groups
    
    def submit_pending_task(self, task_id: str, task_type: str, submit_callback: Callable, 
                           config: Dict[str, Any] = None, save_result_callback: Optional[Callable] = None):
        """
        Submit a task to pending queue (will be executed when resources available)
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task
            submit_callback: Function to call to create futures when resources available
            config: Task configuration
            save_result_callback: Optional function to call when task completes successfully
        
        Returns:
            Dict with submission status
        """
        # Check if already exists and is pending or running
        if task_id in self.tasks:
            existing_task = self.tasks[task_id]
            if existing_task.status in ["pending", "running"]:
                return {"status": "error", "message": f"Task {task_id} already exists (status: {existing_task.status})"}
            # If task is completed or failed, allow resubmission by removing the old task
            # Remove from task_order if present
            if task_id in self.task_order:
                self.task_order.remove(task_id)
            # Remove the old task
            del self.tasks[task_id]
        
        # Create pending task
        task = Task(
            task_id=task_id,
            task_type=task_type,
            config=config or {},
            status="pending",
            submit_callback=submit_callback,
            save_result_callback=save_result_callback
        )
        self.tasks[task_id] = task
        self.task_order.append(task_id)
        
        # Try to schedule immediately
        self._schedule_pending_tasks()
        
        return {
            "status": self.tasks[task_id].status,
            "task_id": task_id,
            "queue_position": self._get_queue_position(task_id)
        }
    
    def register_futures_task(self, task_id: str, task_type: str, futures: List[Any], config: Dict[str, Any] = None):
        """Register a task with already-created futures (directly to running state)"""
        if task_id in self.tasks:
            # Update existing task
            task = self.tasks[task_id]
            task.futures = futures
            task.status = "running"
            task.start_time = time.time()
        else:
            # Create new running task
            task = Task(
                task_id=task_id,
                task_type=task_type,
                config=config or {},
                status="running",
                futures=futures,
                start_time=time.time()
            )
            self.tasks[task_id] = task
            self.task_order.append(task_id)
        
        return {"status": "registered", "task_id": task_id, "num_futures": len(futures)}
    
    def _schedule_pending_tasks(self):
        """Attempt to schedule pending tasks if resources are available"""
        # Count running tasks
        running_count = sum(1 for t in self.tasks.values() if t.status == "running")
        
        # Try to start pending tasks in FIFO order
        for task_id in self.task_order:
            
            task = self.tasks.get(task_id)
            if not task or task.status != "pending":
                continue
            # print(task)
            try:
                # Call the submit callback to get futures
                if task.submit_callback:
                    futures = task.submit_callback()
                    
                    if futures is not None:
                        # Resources ready, got futures - start running
                        task.futures = futures
                        task.status = "running"
                        task.start_time = time.time()
                        running_count += 1
                        task.error = None  # Clear any previous waiting message
                        print(f"✅ Task {task_id} started (attempt {task.retry_count + 1})")
                    else:
                        # Callback returned None - resources not ready yet
                        # Keep task in pending, will check again in next cycle
                        if not task.error or "Waiting" not in task.error:
                            print(f"⏳ Task {task_id} waiting for resources (placement group not ready)")
                            task.error = "Waiting for placement group resources"
                        # Continue to next task (don't break, other tasks might be ready)
                        continue
                        
            except Exception as e:
                # Any exception during callback - fail the task
                import traceback
                task.status = "failed"
                task.error = f"Failed to start: {str(e)}"
                task.result = {"traceback": traceback.format_exc()}
                task.end_time = time.time()
                print(f"❌ Task {task_id} failed: {e} \n{traceback.format_exc()}")
    
    def _get_queue_position(self, task_id: str) -> Optional[int]:
        """Get position of task in pending queue (1-indexed)"""
        position = 0
        for tid in self.task_order:
            task = self.tasks.get(tid)
            if task and task.status == "pending":
                position += 1
                if tid == task_id:
                    return position
        return None
    
    def _cleanup_placement_group(self, task_id: str):
        """Clean up placement group associated with a task"""
        if self.ray_cluster_manager:
            try:
                self.ray_cluster_manager.remove_placement_group(task_id)
            except Exception as e:
                print(f"⚠️  Warning: Failed to cleanup placement group for {task_id}: {e}")
    
    def check_futures_completion(self, task_id: str, timeout: float = 0.01) -> Dict[str, Any]:
        """Check if a futures-based task is completed"""
        task = self.tasks.get(task_id)
        if not task or task.status != "running":
            return {"error": "Task not found or not running"}
        
        # Check if all futures are ready (non-blocking with timeout=0)
        ready_futures, remaining_futures = ray.wait(task.futures, num_returns=len(task.futures), timeout=timeout)

        # print(ready_futures)
        
        if len(ready_futures) == len(task.futures):
            # All futures completed, collect results
            try:
                results = ray.get(ready_futures)
                task.end_time = time.time()
                
                # Check if any subtask failed
                failed_tasks = []
                successful_tasks = []
                for i, result in enumerate(results):
                    if isinstance(result, dict) and result.get("success") == False:
                        failed_tasks.append({
                            "index": i,
                            "error": result.get("error", "Unknown error"),
                            "traceback": result.get("traceback", ""),
                            "rank": result.get("rank", i)
                        })
                    else:
                        successful_tasks.append(result)
                
                if failed_tasks:
                    # Some tasks failed - check if we should retry
                    task.retry_count += 1
                    task.error = f"{len(failed_tasks)}/{len(results)} subtasks failed (attempt {task.retry_count}/{task.max_retries})"
                    task.result = {
                        "num_tasks": len(results),
                        "successful": len(successful_tasks),
                        "failed": len(failed_tasks),
                        "failed_details": failed_tasks,
                        "successful_results": successful_tasks
                    }
                    
                    if task.retry_count < task.max_retries and task.submit_callback:
                        # Retry: move back to pending
                        task.status = "pending"
                        task.futures = None
                        task.end_time = None
                        print(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})...")
                        # Clean up placement group before retry
                        self._cleanup_placement_group(task_id)
                    else:
                        # Max retries reached or no callback
                        task.status = "failed"
                        if task.retry_count >= task.max_retries:
                            task.error += " - Max retries reached"
                        # Clean up placement group for final failure
                        self._cleanup_placement_group(task_id)
                else:
                    # All tasks succeeded
                    task.status = "completed"
                    task.result = {
                        "num_tasks": len(results),
                        "results": results,
                        "all_successful": True
                    }
                    # Save results if callback provided
                    if task.save_result_callback:
                        try:
                            task.save_result_callback(results)
                        except Exception as e:
                            print(f"⚠️  Error in save_result_callback for task {task_id}: {e}")
                    # Clean up placement group after successful completion
                    self._cleanup_placement_group(task_id)
                
                # Task completed/failed, try to schedule pending tasks
                self._schedule_pending_tasks()
                
                return {
                    "status": task.status,
                    "task_id": task_id,
                    "result": task.result,
                    "error": task.error,
                    "elapsed": task.end_time - task.start_time
                }
            except Exception as e:
                # Ray-level exception (e.g., worker crashed)
                import traceback
                task.retry_count += 1
                task.end_time = time.time()
                task.error = f"Ray execution error: {str(e)} (attempt {task.retry_count}/{task.max_retries})"
                task.result = {
                    "traceback": traceback.format_exc()
                }
                
                if task.retry_count < task.max_retries and task.submit_callback:
                    # Retry: move back to pending
                    task.status = "pending"
                    task.futures = None
                    task.end_time = None
                    print(f"Task {task_id} crashed, retrying ({task.retry_count}/{task.max_retries})...")
                    
                    # Clean up placement group before retry
                    self._cleanup_placement_group(task_id)
                    
                    # Task will retry, schedule it
                    self._schedule_pending_tasks()
                    
                    return {
                        "status": "pending",
                        "task_id": task_id,
                        "error": task.error,
                        "retry_count": task.retry_count
                    }
                else:
                    # Max retries reached or no callback
                    task.status = "failed"
                    if task.retry_count >= task.max_retries:
                        task.error += " - Max retries reached"
                    
                    # Clean up placement group for final failure
                    self._cleanup_placement_group(task_id)
                    
                    # Task failed, try to schedule other pending tasks
                    self._schedule_pending_tasks()
                    
                    return {
                        "status": "failed",
                        "task_id": task_id,
                        "error": task.error,
                        "result": task.result,
                        "elapsed": task.end_time - task.start_time if task.start_time else 0
                    }
        else:
            # Still running
            results = ray.get(ready_futures)
            # Check if any subtask failed
            failed_tasks = []
            successful_tasks = []
            for i, result in enumerate(results):
                if isinstance(result, dict) and result.get("success") == False:
                    failed_tasks.append({
                        "index": i,
                        "error": result.get("error", "Unknown error"),
                        "traceback": result.get("traceback", ""),
                        "rank": result.get("rank", i)
                    })
                else:
                    successful_tasks.append(result)
            
            if failed_tasks:
                print(f"❌ Task {task_id} subtasks failed: {failed_tasks}")
            elapsed = time.time() - task.start_time if task.start_time else 0
            return {
                "status": "running",
                "task_id": task_id,
                "completed": len(ready_futures),
                "total": len(task.futures),
                "elapsed": elapsed
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        # Try to schedule pending tasks when querying status
        self._schedule_pending_tasks()
        
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}
        
        # For running tasks, check futures completion
        if task.status == "running":
            return self.check_futures_completion(task_id)
        
        # For pending tasks
        if task.status == "pending":
            return {
                "task_id": task.task_id,
                "status": "pending",
                "queue_position": self._get_queue_position(task_id),
                "wait_time": time.time() - task.create_time
            }
        
        # For completed/failed tasks
        return {
            "task_id": task.task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "elapsed": task.end_time - task.start_time if task.end_time and task.start_time else 0
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks (pending, running, completed)"""
        # Try to schedule pending tasks when querying status
        self._schedule_pending_tasks()
        
        result = []
        
        for task in self.tasks.values():
            task_info = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status,
            }
            
            if task.status == "pending":
                task_info["queue_position"] = self._get_queue_position(task.task_id)
                task_info["wait_time"] = time.time() - task.create_time
                if task.error:
                    task_info["error"] = task.error
            
            elif task.status == "running":
                # Check if task is actually completed (this will update status if done)
                completion_status = self.check_futures_completion(task.task_id)
                
                # Refresh task reference in case status changed
                task = self.tasks.get(task.task_id)
                if not task:
                    continue
                
                # Update task_info with current status
                task_info["status"] = task.status
                
                if task.status in ["completed", "failed"]:
                    # Task completed or failed (status was updated by check_futures_completion)
                    if task.end_time and task.start_time:
                        task_info["elapsed"] = task.end_time - task.start_time
                    if task.error:
                        task_info["error"] = task.error
                    if task.result and "node_info" in task.result:
                        task_info["node_info"] = task.result["node_info"]
                elif completion_status.get("status") == "running":
                    # Still running, use progress info from check_futures_completion
                    if "elapsed" in completion_status:
                        task_info["elapsed"] = completion_status["elapsed"]
                    if "completed" in completion_status and "total" in completion_status:
                        task_info["completed_subtasks"] = completion_status["completed"]
                        task_info["total_subtasks"] = completion_status["total"]
                    
                    if task.result and "node_info" in task.result:
                        task_info["node_info"] = task.result["node_info"]
                else:
                    # Fallback: show basic running info
                    if task.start_time:
                        task_info["elapsed"] = time.time() - task.start_time
                    if task.futures:
                        task_info["total_subtasks"] = len(task.futures)
            
            else:  # completed or failed
                if task.end_time and task.start_time:
                    task_info["elapsed"] = task.end_time - task.start_time
                if task.error:
                    task_info["error"] = task.error
                if task.result and "node_info" in task.result:
                    task_info["node_info"] = task.result["node_info"]
            
            result.append(task_info)
        
        return result
    
    def cancel_task(self, task_id: str) -> str:
        """Cancel a task (only pending tasks can be cancelled)"""
        task = self.tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found"
        
        if task.status == "pending":
            # Clean up any placement group that may have been created
            self._cleanup_placement_group(task_id)
            # Remove from tasks
            del self.tasks[task_id]
            self.task_order.remove(task_id)
            return f"Task {task_id} cancelled (was pending in queue)"
        
        if task.status == "running":
            return f"Cannot cancel running task {task_id}"
        
        return f"Task {task_id} already {task.status}"
    
    def retry_task(self, task_id: str) -> str:
        """Manually retry a failed task"""
        task = self.tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found"
        
        if task.status != "failed":
            return f"Task {task_id} is not failed (status: {task.status})"
        
        if not task.submit_callback:
            return f"Task {task_id} has no submit_callback, cannot retry"
        
        # Reset task to pending
        task.status = "pending"
        task.futures = None
        task.start_time = None
        task.end_time = None
        task.retry_count = 0  # Reset retry count for manual retry
        task.error = None
        task.result = None
        
        # Try to schedule
        self._schedule_pending_tasks()
        
        return f"Task {task_id} queued for retry"
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        # Try to schedule pending tasks when querying stats
        self._schedule_pending_tasks()
        
        pending = sum(1 for t in self.tasks.values() if t.status == "pending")
        running = sum(1 for t in self.tasks.values() if t.status == "running")
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        
        return {
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
        }

