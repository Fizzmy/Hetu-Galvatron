"""Base class for all profile managers"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray

class BaseProfileManager(ABC):
    """Base class for hardware profile managers"""
    
    def __init__(self, config_file: Path):
        """
        Initialize hardware profile manager
        
        Args:
            config_file: Path to the configuration file for storing profiling results
        """
        self.config_file = config_file
        self.data: Dict[str, Any] = {}
        self._load_data()
    
    def _load_data(self):
        """Load existing keys from config file, create file if it doesn't exist"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.data = json.load(f)
            except:
                # If file exists but is corrupted, create empty file
                self.data = {}
                self._save_data()
        else:
            # Create file and parent directories if they don't exist
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.data = {}
            self._save_data()
    
    def _save_data(self):
        """Save data to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            # Silently fail if we can't save (e.g., permission issues)
            pass
    
    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """
        Get list of required keys that must be present for profiling to be considered completed
        
        Returns:
            List of required key names
        """
        pass
    
    def is_completed(self) -> bool:
        """Check if profiling is completed by verifying all required keys exist"""
        required_keys = self.get_required_keys()
        for key in required_keys:
            if key not in self.data:
                return False
        return True
    
    def get_missing_keys(self) -> List[str]:
        """Get list of missing keys that need to be profiled"""
        required_keys = self.get_required_keys()
        return [key for key in required_keys if key not in self.data]
    
    def get_keys_to_profile(self, profile_missing_only: bool = False) -> Optional[List[str]]:
        """Get list of keys to profile"""
        if profile_missing_only:
            return self.get_missing_keys()
        else:
            return self.get_required_keys()
    
    def save_result(self, results: List[Dict[str, Any]]) -> bool:
        """
        Save profiling results to file
        
        Args:
            results: List of results from all ranks
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Extract data from rank 0
            for res in results:
                if 'data' in res:
                    # Save all key-value pairs from rank 0's data
                    for key, value in res['data'].items():
                        if key not in self.data:
                            self.data[key] = {}
                        if "bsz" in key: # memory or computation
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    self.data[key][k] = v
                            elif value is not None:
                                self.data[key] = value
                        elif res.get('rank') == 0:
                            self.data[key] = value
            
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.config_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            
            print(f"✅ Saved profiling results to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving profiling results: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_submit_callback(self, task_id: str, args: Any):
        """
        Create a callback function for submitting profiling task
        
        Args:
            task_id: Unique task identifier for tracking placement group
            args: Profiling arguments (dataclass instance)
        
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
            _master_addr, _master_port = state.ray_cluster.get_master_addr_port(pg)
            futures = self.execute(pg, _master_addr, _master_port, args, self.total_gpus)
            return futures
        
        return submit_callback
    
    def create_save_result_callback(self):
        """
        Create a callback function for saving profiling results
        
        Returns:
            Callable that saves results to file
        """
        def save_result_callback(results: List[Dict[str, Any]]):
            self.save_result(results)
        
        return save_result_callback
    
    @abstractmethod
    def get_profile_function(self):
        """
        Get the Ray remote function for profiling
        
        Returns:
            Ray remote function to be used for profiling
        """
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        """
        Get the task type identifier
        
        Returns:
            Task type string (e.g., "profile_overlap", "profile_allreduce")
        """
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """
        Get the task description for display
        
        Returns:
            Task description string
        """
        pass
    
    @abstractmethod
    def get_args_for_key(self, key: Optional[str] = None) -> Any:
        """
        Get arguments for profiling a specific key
        
        Args:
            key: Key to profile (None for single-task profiling like overlap)
        
        Returns:
            Arguments object (dataclass instance) that can be used with attribute access
        """
        pass
    
    def execute(self, pg: PlacementGroup, _master_addr: str, _master_port: str, args: Any, gpu_nums: int):
        """
        Execute profiling asynchronously, returns list of futures
        
        Args:
            pg: Ray placement group
            _master_addr: Master address for distributed training
            _master_port: Master port for distributed training
            args: Profiling arguments (dataclass instance)
            gpus_per_node: Number of GPUs per node
        
        Returns:
            List of Ray futures
        """
        # Load additional environment variables from gui_config.json
        PROJECT_ROOT = Path(__file__).parent.parent
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
            "MASTER_ADDR": _master_addr,
            "MASTER_PORT": _master_port,
        }
        # Merge with config vars (config vars take precedence)
        base_env_vars.update(ray_env_vars)
        
        futures = []
        remote_profile_func = ray.remote(num_gpus=1)(self.get_profile_function())
        for i in range(gpu_nums):
            # Create env_vars for this GPU (copy base and add GPU-specific vars)
            env_vars = base_env_vars.copy()
            env_vars["RANK"] = str(i)
            env_vars["LOCAL_RANK"] = str(i % gpu_nums)
            
            future = remote_profile_func.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(args)
            futures.append(future)
        return futures
    
    def submit_task(self, profile_missing_only: bool = False) -> str:
        """
        Submit profiling task(s) to pending queue (unified method for all profile types)
        
        Args:
            profile_missing_only: If True, only profile missing data keys (for multi-key profiling)
        
        Returns:
            Status message string
        """
        from ui.state import state
        
        if not state.ray_task_manager:
            return "Ray cluster not initialized. Please initialize Ray first."
        
        try:
            task_type = self.get_task_type()
            description = self.get_task_description()
            keys_to_profile = self.get_keys_to_profile(profile_missing_only)

            print("keys_to_profile: ", keys_to_profile)
            
            # Check if keys_to_profile is empty
            if not keys_to_profile:
                return f"✅ All data already profiled. No missing data to profile."
            
            # Unified loop: create one task for each key
            submitted_tasks = []
            failed_tasks = []
            
            for idx, key in enumerate(keys_to_profile):
                # Generate task_id: include key and index
                task_id = f"{task_type}_{key}"
                
                # Get args for this key (dataclass instance)
                args = self.get_args_for_key(key)
                
                # Create submit callback with task_id for placement group tracking
                submit_callback = self.create_submit_callback(task_id, args)
                
                # Create save result callback
                save_result_callback = self.create_save_result_callback()
                
                # Build config: store args as dict for serialization, but pass object to callback
                config = {"args": args, "profile_key": key}
                print("config: ", config)
                # Submit to pending queue
                result = state.ray_task_manager.submit_pending_task(
                    task_id=task_id,
                    task_type=task_type,
                    submit_callback=submit_callback,
                    config=config,
                    save_result_callback=save_result_callback
                )
                
                if result.get("status") == "error":
                    failed_tasks.append((key, result.get('message')))
                else:
                    submitted_tasks.append(key)
            
            # Build summary message
            if failed_tasks:
                failed_msg = "\n".join([f"  - `{key}`: {msg}" for key, msg in failed_tasks])
                return f"""⚠️ Submitted {len(submitted_tasks)}/{len(keys_to_profile)} tasks for {description}

**Successfully Queued:** {len(submitted_tasks)} tasks
**Failed:** {len(failed_tasks)} tasks
{failed_msg}

📊 **Next:** Go to 'Task Monitor' tab to view all tasks
"""
            else:
                key_type = "missing" if profile_missing_only else "all required"
                return f"""✅ Submitted {len(submitted_tasks)} tasks for {description}

**Tasks Created:** {len(submitted_tasks)} (one per {key_type} key)

📊 **Next:** Go to 'Task Monitor' tab to view all tasks
💡 **Tip:** Each key has its own task for independent profiling
"""
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
