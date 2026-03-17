"""Task Monitor event handlers"""

from ..state import state


def refresh_task_list():
    """Refresh task list"""
    if not state.ray_task_manager:
        return "Ray not initialized"
    
    try:
        tasks = state.ray_task_manager.get_all_tasks()
        stats = state.ray_task_manager.get_queue_stats()
        
        if not tasks:
            return "No tasks found"
        
        output = f"**Queue Statistics:**\n"
        output += f"- 🕐 Pending: {stats['pending']}\n"
        output += f"- ▶️ Running: {stats['running']}\n"
        output += f"- ✅ Completed: {stats['completed']}\n"
        output += f"- ❌ Failed: {stats['failed']}\n\n"
        
        output += "**All Tasks:**\n\n"
        
        # Sort: pending first, then running, then completed
        status_order = {"pending": 0, "running": 1, "completed": 2, "failed": 2}
        for task in sorted(tasks, key=lambda x: (status_order.get(x.get('status'), 99), x.get('task_id'))):
            status_emoji = {
                "pending": "🕐",
                "running": "▶️",
                "completed": "✅",
                "failed": "❌"
            }.get(task['status'], "❓")
            
            output += f"**{status_emoji} {task['task_id']}**\n"
            output += f"- Type: {task['task_type']}\n"
            output += f"- Status: {task['status']}\n"
            
            # Pending task info
            if task['status'] == 'pending':
                if 'queue_position' in task:
                    output += f"- Queue Position: #{task['queue_position']}\n"
                if 'wait_time' in task:
                    output += f"- Waiting: {task['wait_time']:.1f}s\n"
            
            # Running task info
            if task['status'] == 'running':
                if 'elapsed' in task:
                    output += f"- Elapsed: {task['elapsed']:.1f}s\n"
                
                # Show progress for futures-based tasks
                if 'completed_subtasks' in task and 'total_subtasks' in task:
                    completed = task['completed_subtasks']
                    total = task['total_subtasks']
                    progress = (completed / total * 100) if total > 0 else 0
                    output += f"- Progress: {completed}/{total} ({progress:.0f}%)\n"
            
            # Completed task info
            if task['status'] in ['completed', 'failed']:
                if 'elapsed' in task:
                    output += f"- Elapsed: {task['elapsed']:.1f}s\n"
            
            if 'node_info' in task and task['node_info']:
                output += f"- Node: {task['node_info'].get('hostname', 'unknown')} ({task['node_info'].get('node_ip', 'unknown')})\n"
            
            # Show error details for failed tasks
            if task['status'] == 'failed' and task.get('error'):
                output += f"- ❌ Error: {task['error']}\n"
                
                # Get detailed task status for more error info
                task_status = state.ray_task_manager.get_task_status(task['task_id'])
                if task_status.get('result') and 'failed_details' in task_status['result']:
                    failed_details = task_status['result']['failed_details']
                    output += f"- Failed subtasks: {len(failed_details)}\n"
                    # Show all failed subtasks; wrap traceback in code block to avoid truncation in UI
                    for failed in failed_details:
                        rank = failed.get('rank', 'unknown')
                        err = failed.get('error', 'unknown error')
                        tb = (failed.get('traceback') or "").strip()
                        output += f"  - Rank {rank}: {err}\n"
                        if tb:
                            output += f"```\n{tb}\n```\n"
                        else:
                            output += "(no traceback)\n"
            
            output += "\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


def cancel_task_by_id(task_id: str):
    """Cancel a task"""
    if not state.ray_task_manager:
        return "Ray not initialized"
    
    try:
        result = state.ray_task_manager.cancel_task(task_id)
        return f"✅ {result}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def retry_task_by_id(task_id: str):
    """Retry a failed task"""
    if not state.ray_task_manager:
        return "Ray not initialized"
    
    if not task_id or not task_id.strip():
        return "Please enter a task ID"
    
    try:
        result = state.ray_task_manager.retry_task(task_id.strip())
        return f"✅ {result}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

