"""Application state management"""

from core import RayClusterManager
from profiling import OverlapCoefficientManager, BandwidthProfileManager, ModelProfileManager
from profiling.search import SearchManager
from profiling.training import TrainingManager
from profiling.arguments import SearchArgs, TrainingArgs, ModelConfigs


class AppState:
    """Application state manager"""
    
    def __init__(self):
        self.ray_cluster = RayClusterManager()
        self.ray_task_manager = None
        self.current_hw_config = None  # Deprecated, kept for compatibility
        self.current_hw_config_nodes = None
        self.current_hw_config_gpus = None
        self.cluster_info = None
        self.overlap_manager = None
        # Bandwidth managers stored by config key: (num_nodes, num_gpus_per_node)
        self.bandwidth_managers = {}  # {(num_nodes, num_gpus_per_node): {dp, tp, pp, sp}}
        # Model configuration cache: {model_type: [model_size1, model_size2, ...]}
        # e.g., {"llama": ["7b", "13b", "30b"], "gpt": ["7b", "13b"]}
        self.model_config_cache = {}
        # Model profile managers stored by key: (model_identifier, profile_type, profile_mode)
        # e.g., {("llama_7b", "computation", "static"): ModelProfileManager(...)}
        self.model_profile_managers = {}  # {(model_identifier, profile_type, profile_mode): ModelProfileManager}
        # Search managers stored by key: (model_identifier, cluster_config)
        # e.g., {("llama-7b", "1_8"): SearchManager(...)}
        self.search_managers = {}  # {(model_identifier, cluster_config): SearchManager}
        # Training managers stored by key: (model_identifier, cluster_config, galvatron_config_path)
        # e.g., {("llama-7b", "1_8", "/path/to/config.json"): TrainingManager(...)}
        self.training_managers = {}  # {(model_identifier, cluster_config, galvatron_config_path): TrainingManager}
    
    def get_or_create_bandwidth_managers(self, num_nodes: int, num_gpus_per_node: int):
        """Get or create bandwidth managers for a specific configuration"""
        config_key = (num_nodes, num_gpus_per_node)
        if config_key not in self.bandwidth_managers:
            self.bandwidth_managers[config_key] = {
                'dp': BandwidthProfileManager(num_nodes, num_gpus_per_node, "allreduce", "dp"),
                'tp': BandwidthProfileManager(num_nodes, num_gpus_per_node, "allreduce", "tp"),
                'pp': BandwidthProfileManager(num_nodes, num_gpus_per_node, "p2p", "pp"),
                'sp': BandwidthProfileManager(num_nodes, num_gpus_per_node, "all2all", "sp"),
            }
        return self.bandwidth_managers[config_key]
    
    def init_hardware_profile_managers(self):
        """Initialize overlap manager when cluster info is available"""
        if self.cluster_info:
            self.overlap_manager = OverlapCoefficientManager(self.cluster_info['gpus_per_node'])
    
    def get_or_create_model_profile_manager(
        self, model_identifier: str, profile_type: str, profile_mode: str,
        profile_args, model_args
    ):
        """Get or create model profile manager for a specific configuration
        
        Args:
            model_identifier: Model identifier (e.g., 'llama-7b')
            profile_type: Profile type ('computation' or 'memory')
            profile_mode: Profile mode ('static', 'batch', or 'sequence')
            profile_args: ModelProfileConfigs instance
            model_args: ModelConfigs instance
        
        Returns:
            ModelProfileManager instance
        """
        key = (model_identifier, profile_type, profile_mode)
        self.model_profile_managers[key] = ModelProfileManager(
            self.cluster_info['gpus_per_node'], model_identifier, profile_type, profile_mode,
            profile_args, model_args
        )
        return self.model_profile_managers[key]
    
    def get_or_create_search_manager(
        self, search_args: SearchArgs, model_configs: ModelConfigs,
    ):
        """Get or create search manager for a specific configuration
        
        Args:
            search_args: Search arguments dataclass
            model_configs: Model configuration dataclass
        
        Returns:
            SearchManager instance
        """
        key = (search_args.model_identifier, f"{search_args.num_nodes}_{search_args.num_gpus_per_node}")
        self.search_managers[key] = SearchManager(search_args=search_args, model_configs=model_configs)
        return self.search_managers[key]
    
    def get_or_create_training_manager(
        self, training_args: TrainingArgs, model_configs: ModelConfigs,
    ):
        """Get or create training manager for a specific configuration
        
        Args:
            training_args: Training arguments dataclass
            model_configs: Model configuration dataclass
        
        Returns:
            TrainingManager instance
        """
        key = (training_args.model_identifier, f"{training_args.num_nodes}_{training_args.num_gpus_per_node}", training_args.galvatron_config_path)
        self.training_managers[key] = TrainingManager(training_args=training_args, model_configs=model_configs)
        return self.training_managers[key]

# Global state instance
state = AppState()

