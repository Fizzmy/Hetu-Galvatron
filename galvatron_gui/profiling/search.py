"""Strategy search management"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .arguments import SearchArgs, ModelConfigs
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

class SearchManager:
    """Manages parallelism strategy search tasks"""
    
    def __init__(
        self, search_args: SearchArgs, model_configs: ModelConfigs,
    ):
        """
        Initialize search manager
        
        Args:
            search_args: Search arguments dataclass
            model_configs: Model configuration dataclass
        """
        self.search_args = search_args
        self.model_configs = model_configs
        
        # Calculate total CPUs for parallel search
        # Search uses multiple CPUs for multi-threaded parallel exploration
        # Default to 4 CPUs if cannot detect, limit to 16 for reasonable resource usage
        self.total_cpus = min(os.cpu_count() or 4, 16)  # Use CPUs for multi-threading

    def get_task_type(self) -> str:
        """Get the task type identifier"""
        return f"search_{self.search_args.model_identifier}"
    
    def get_task_description(self) -> str:
        """Get the task description"""
        return f"Parallelism Strategy Search for {self.search_args.model_identifier}"
    
    def _get_args(self) -> Any:
        """Get arguments for search task
        
        Returns:
            Combined args object with search_args and model_configs merged
        """
        # Start with search_args as base
        # Create a copy to avoid modifying the original
        import copy
        args = copy.deepcopy(self.search_args)
        
        # Merge model_configs into args
        for k, v in self.model_configs.__dict__.items():
            if v is not None:  # Only set non-None values
                setattr(args, k, v)
        
        args.parallel_search = True
        args.worker = self.total_cpus
        return args
    
    def _create_submit_callback(self, task_id: str, args: Any):
        """
        Create a callback function for submitting search task using CPUs
        
        Args:
            task_id: Unique task identifier for tracking placement group
            args: Combined args object
        
        Returns:
            Callable that returns futures when resources ready, or None if waiting
        """
        def submit_callback():
            from ui.state import state
            
            # Check if Ray cluster is initialized
            if not state.ray_cluster:
                raise RuntimeError("Ray cluster not initialized")
            
            # Try to create placement group if not exists (non-blocking)
            # This will return immediately, but the group may not be ready yet
            state.ray_cluster.create_placement_group_for_task(task_id, num_cpus=self.total_cpus)
            
            # Non-blocking check: Is placement group ready?
            # Returns None if not ready (continue waiting), or pg if ready
            pg = state.ray_cluster.check_placement_group_ready(task_id)
            
            if pg is None:
                # Resources not ready yet, return None to keep task pending
                # Scheduler will retry later
                return None
            
            # Resources ready, proceed to create futures
            futures = self.execute(pg, args, self.total_cpus)
            return futures
        
        return submit_callback
    
    def _create_save_result_callback(self):
        """
        Create a callback function for saving search results
        
        Returns:
            Callable that saves results to file
        """
        def save_result_callback(results: List[Dict[str, Any]]):
            self.save_result(results)
        
        return save_result_callback
    
    def save_result(self, results: List[Dict[str, Any]]):
        """
        Save search results to file
        
        Args:
            results: List of search results
        """
        print(f"✅ Search results: {results}")
        return True

    def execute(self, pg: PlacementGroup, args: Any, cpu_nums: int):
        """
        Execute search asynchronously, returns list of futures
        
        Args:
            pg: Ray placement group
            args: Search arguments (dataclass instance)
            cpu_nums: Number of CPUs per node
        
        Returns:
            List of Ray futures
        """
        futures = []
        from galvatron.models.llama_hf.search_dist import search_remote
        remote_search_func = ray.remote(num_cpus=cpu_nums)(search_remote)
        future = remote_search_func.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
            runtime_env={"env_vars": {
                "LAUNCH_BACKEND": "ray",
            }},
        ).remote(args)
        futures.append(future)
        return futures

    def submit_task(self) -> str:
        """
        Submit search task to pending queue
        
        Args:
            profile_missing_only: Not used for search (always submits one task)
        
        Returns:
            Status message string
        """
        from ui.state import state
        
        if not state.ray_task_manager:
            return "Ray cluster not initialized. Please initialize Ray first."
        
        try:
            task_type = self.get_task_type()
            description = self.get_task_description()
            
            # Search is always a single task
            task_id = task_type
            
            # Get args for search
            args = self._get_args()
            
            # Create submit callback
            submit_callback = self._create_submit_callback(task_id, args)
            
            # Create save result callback
            save_result_callback = self._create_save_result_callback()
            
            # Build config: store args as dict for serialization
            config = {"args": args}
            
            # Submit to pending queue
            result = state.ray_task_manager.submit_pending_task(
                task_id=task_id,
                task_type=task_type,
                submit_callback=submit_callback,
                config=config,
                save_result_callback=save_result_callback
            )
            
            if result.get("status") == "error":
                return f"❌ Error submitting search task: {result.get('message')}"
            else:
                return f"""✅ Search task submitted successfully!

**Task ID:** `{task_id}`
**Description:** {description}

📊 **Next:** Go to 'Task Monitor' tab to view task progress
"""
        except Exception as e:
            import traceback
            return f"❌ Error submitting search task: {str(e)}\n\n{traceback.format_exc()}"
