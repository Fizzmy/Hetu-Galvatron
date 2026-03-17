"""Distributed training management"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .arguments import TrainingArgs, ModelConfigs
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
GALVATRON_ROOT = Path(PROJECT_ROOT.parent) / "galvatron"
TRAINING_LOGS_DIR = PROJECT_ROOT / "data" / "training_logs"
TRAINING_LOGS_DIR.mkdir(parents=True, exist_ok=True)


class TrainingManager:
    """Manages distributed training tasks"""
    
    def __init__(
        self, training_args: TrainingArgs, model_configs: ModelConfigs,
    ):
        """
        Initialize training manager
        
        Args:
            training_args: Training arguments dataclass
            model_configs: Model configuration dataclass
        """
        self.training_args = training_args
        self.model_configs = model_configs
        
        # Training uses GPUs (not CPUs like search)
        self.total_gpus = training_args.num_nodes * training_args.num_gpus_per_node
        
        # Generate log directory and file name
        model_identifier = training_args.model_identifier
        # Create a directory for this training task's logs (without timestamp)
        self.log_dir = TRAINING_LOGS_DIR / f"training_{model_identifier}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_task_type(self) -> str:
        """Get the task type identifier"""
        return f"training_{self.training_args.model_identifier}"
    
    def get_task_description(self) -> str:
        """Get the task description"""
        return f"Distributed Training for {self.training_args.model_identifier}"
    
    def _get_args(self) -> Any:
        """Get arguments for training task
        
        Returns:
            Combined args object with training_args and model_configs merged
        """
        # Start with training_args as base
        # Create a copy to avoid modifying the original
        import copy
        args = copy.deepcopy(self.training_args)
        
        # Merge model_configs into args
        for k, v in self.model_configs.__dict__.items():
            if v is not None:  # Only set non-None values
                setattr(args, k, v)
        
        # Debug: Print all attributes using __dict__.items()
        # Note: dataclass __repr__ only shows declared fields, but setattr works fine
        print("=== All args attributes (including model_configs) ===")
        for key, value in sorted(args.__dict__.items()):
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        # Note: log_dir will be set per rank in execute() method
        # since we need to know the actual rank number from the loop
        return args
    
    def _create_submit_callback(self, task_id: str, args: Any):
        """
        Create a callback function for submitting training task using GPUs
        
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
            state.ray_cluster.create_placement_group_for_task(task_id, num_gpus=self.total_gpus)
            
            # Non-blocking check: Is placement group ready?
            # Returns None if not ready (continue waiting), or pg if ready
            pg = state.ray_cluster.check_placement_group_ready(task_id)
            
            if pg is None:
                # Resources not ready yet, return None to keep task pending
                # Scheduler will retry later
                return None
            
            # Resources ready, proceed to create futures
            # Get master address and port for distributed training
            master_addr, master_port = state.ray_cluster.get_master_addr_port(pg)
            futures = self.execute(pg, master_addr, master_port, args, self.total_gpus)
            return futures
        
        return submit_callback
    
    def _create_save_result_callback(self):
        """
        Create a callback function for saving training results
        
        Returns:
            Callable that saves results to file
        """
        def save_result_callback(results: List[Dict[str, Any]]):
            self.save_result(results)
        
        return save_result_callback
    
    def save_result(self, results: List[Dict[str, Any]]):
        """
        Save training results to file
        
        Args:
            results: List of training results from all ranks
        """
        print(f"✅ Training completed: {results}")
        return True

    def execute(self, pg: PlacementGroup, master_addr: str, master_port: str, args: Any, gpu_nums: int):
        """
        Execute training asynchronously, returns list of futures
        
        Args:
            pg: Ray placement group
            master_addr: Master address for distributed training
            master_port: Master port for distributed training
            args: Training arguments (dataclass instance)
            gpu_nums: Number of GPUs
        
        Returns:
            List of Ray futures
        """
        # Load additional environment variables from gui_config.json
        config_path = PROJECT_ROOT / "data" / "gui_config.json"
        ray_env_vars = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    ray_env_vars = config.get("ray_env_vars", {})
            except Exception:
                pass
        
        # Base env vars (will be updated per GPU)
        base_env_vars = {
            "WORLD_SIZE": str(gpu_nums),
            "LAUNCH_BACKEND": "ray",
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        }
        # Merge with config vars (config vars take precedence)
        base_env_vars.update(ray_env_vars)

        print(base_env_vars)
        
        futures = []
        from galvatron.models.llama_hf.train_dist import train_remote
        remote_train_func = ray.remote(num_gpus=1)(train_remote)
        for i in range(gpu_nums):
            # Create env_vars for this GPU (copy base and add GPU-specific vars)
            env_vars = base_env_vars.copy()
            env_vars["RANK"] = str(i)
            env_vars["LOCAL_RANK"] = str(i % gpu_nums)
            
            # Create log directory and file for this rank
            log_file_path = self.log_dir / f"training_rank_{i}.log"
            
            # Set log_dir in args for this specific rank
            # Create a copy of args to avoid modifying the original
            import copy
            rank_args = copy.deepcopy(args)
            rank_args.log_dir = str(log_file_path)
            rank_args.rank = i  # Also set rank for reference
            
            future = remote_train_func.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i
                ),
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(rank_args)
            futures.append(future)
        return futures

    def submit_task(self) -> str:
        """
        Submit training task to pending queue
        
        Returns:
            Status message string
        """
        from ui.state import state
        
        if not state.ray_task_manager:
            return "Ray cluster not initialized. Please initialize Ray first."
        
        try:
            task_type = self.get_task_type()
            description = self.get_task_description()
            
            # Training is always a single task
            task_id = task_type
            
            # Get args for training
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
                return f"❌ Error submitting training task: {result.get('message')}"
            else:
                return f"""✅ Training task submitted successfully!

**Task ID:** `{task_id}`
**Description:** {description}

📊 **Next:** Go to 'Task Monitor' tab to view task progress
"""
        except Exception as e:
            import traceback
            return f"❌ Error submitting training task: {str(e)}\n\n{traceback.format_exc()}"
