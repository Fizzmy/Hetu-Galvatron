"""Hardware profiling management"""

import json
import ray
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from galvatron.profile_hardware.profile_overlap import ray_profile_overlap
from galvatron.profile_hardware.profile_allreduce import ray_profile_allreduce
from galvatron.profile_hardware.profile_p2p import ray_profile_p2p
from galvatron.profile_hardware.profile_all2all import ray_profile_all2all
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from .arguments import OverlapCoefficientArgs, HardwareProfileArgs
from .base import BaseProfileManager

# Get project root (galvatron_gui directory)
PROJECT_ROOT = Path(__file__).parent.parent
HARDWARE_CONFIGS_DIR = PROJECT_ROOT / "data" / "profiling_results" / "hardware"
HARDWARE_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

class OverlapCoefficientManager(BaseProfileManager):
    """Manages overlap coefficient profiling"""
    
    def __init__(self, num_gpus_per_node: int):
        config_file = HARDWARE_CONFIGS_DIR / "overlap_coefficient.json"
        super().__init__(config_file)
        # Get total_gpus from cluster info (gpus_per_node for overlap profiling)
        self.total_gpus = num_gpus_per_node
    
    def get_required_keys(self) -> List[str]:
        """Get required keys for overlap coefficient profiling"""
        return ['overlap_coe']
    
    def get_status(self) -> str:
        """Get status"""
        if self.is_completed():
            return f"✅ Completed - overlap coe: {self.data['overlap_coe']}"
        else:
            return "❌ Not completed - Need profiling"
    
    def get_profile_function(self):
        """Get the Ray remote function for overlap profiling"""
        return ray_profile_overlap
    
    def get_task_type(self) -> str:
        """Get the task type identifier"""
        return "profile_overlap"
    
    def get_task_description(self) -> str:
        """Get the task description"""
        return "Overlap Coefficient Profiling"
    
    def get_keys_to_profile(self, profile_missing_only: bool = False) -> Optional[List[str]]:
        """Get list of keys to profile"""
        if profile_missing_only:
            # Check if overlap_coe is missing
            if 'overlap_coe' not in self.data:
                return ['overlap_coe']
            else:
                return []
        return ['overlap_coe']  # Always return the single key for overlap
    
    def get_args_for_key(self, key: Optional[str] = None) -> Any:
        """Get arguments for overlap profiling"""
        from profiling.arguments import OverlapCoefficientArgs
        args = OverlapCoefficientArgs()
        args.key = key
        return args


class BandwidthProfileManager(BaseProfileManager):
    """Manages bandwidth profiling"""
    
    def __init__(self, num_nodes: int, num_gpus_per_node: int, profile_type: str, parallel_strategy: str):
        # Generate config file name based on profile_type and parallel_strategy
        # Format: {profile_type}_bandwidth_{nodes}nodes_{gpus}gpus_per_node.json
        # For allreduce, use consec0 or consec1 based on strategy
        if profile_type == "allreduce":
            if parallel_strategy == "dp":
                # DP uses consec0 for non-max sizes, consec1 for max size
                # We'll use consec0 file for DP (mixed consec values)
                config_file = HARDWARE_CONFIGS_DIR / f"allreduce_consec0_bandwidth_{num_nodes}nodes_{num_gpus_per_node}gpus_per_node.json"
            elif parallel_strategy == "tp":
                # TP uses consec1
                config_file = HARDWARE_CONFIGS_DIR / f"allreduce_consec1_bandwidth_{num_nodes}nodes_{num_gpus_per_node}gpus_per_node.json"
            else:
                raise ValueError(f"Invalid parallel_strategy for allreduce: {parallel_strategy}")
        else:
            # p2p, all2all use simple format
            config_file = HARDWARE_CONFIGS_DIR / f"{profile_type}_bandwidth_{num_nodes}nodes_{num_gpus_per_node}gpus_per_node.json"
        
        super().__init__(config_file)
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.total_gpus = num_nodes * num_gpus_per_node
        self.profile_type = profile_type
        self.parallel_strategy = parallel_strategy
        self.need_to_profile = []
        if profile_type == "allreduce" and parallel_strategy == "dp":
            max_num = self.total_gpus
            while max_num > 1:
                # DP: consec=1 for max size, consec=0 for others
                self.need_to_profile.append(f"allreduce_size_{max_num}")
                max_num //= 2
        elif profile_type == "allreduce" and parallel_strategy == "tp":
            max_num = self.total_gpus
            while max_num > 1:
                # TP: all consec=1
                self.need_to_profile.append(f"allreduce_size_{max_num}")
                max_num //= 2
        elif profile_type == "p2p" and parallel_strategy == "pp":
            max_num = self.total_gpus
            while max_num > 1:
                self.need_to_profile.append(f"pp_size_{max_num}")
                max_num //= 2
        elif profile_type == "all2all" and parallel_strategy == "sp":
            max_num = self.total_gpus
            while max_num > 1:
                self.need_to_profile.append(f"all2all_size_{max_num}")
                max_num //= 2
        else:
            raise ValueError(f"Invalid profile type or parallel strategy: {profile_type} {parallel_strategy}")
    
    def get_required_keys(self) -> List[str]:
        """Get required keys for bandwidth profiling"""
        return self.need_to_profile
    
    def get_status(self) -> str:
        """Get status"""
        if self.is_completed():
            return f"✅ Completed - {self.profile_type} {self.parallel_strategy} bandwidth: {self.data}"
        else:
            return "❌ Not completed - Need profiling"
    
    def get_profile_function(self):
        """Get the Ray remote function for bandwidth profiling"""
        # TODO: Import and return appropriate profiling function based on profile_type
        # For now, this needs to be implemented based on the actual profiling functions
        if self.profile_type == "allreduce" and self.parallel_strategy == "dp":
            return ray_profile_allreduce
        elif self.profile_type == "allreduce" and self.parallel_strategy == "tp":
            return ray_profile_allreduce
        elif self.profile_type == "p2p" and self.parallel_strategy == "pp":
            return ray_profile_p2p
        elif self.profile_type == "all2all" and self.parallel_strategy == "sp":
            return ray_profile_all2all
        else:
            raise ValueError(f"Unknown profile type or parallel strategy: {self.profile_type} {self.parallel_strategy}")
    
    def get_task_type(self) -> str:
        """Get the task type identifier"""
        if self.profile_type == "allreduce" and self.parallel_strategy == "dp":
            return "profile_allreduce"
        elif self.profile_type == "p2p" and self.parallel_strategy == "pp":
            return "profile_p2p"
        elif self.profile_type == "allreduce" and self.parallel_strategy == "tp":
            return "profile_tp_allreduce"
        elif self.profile_type == "all2all" and self.parallel_strategy == "sp":
            return "profile_sp_all2all"
        else:
            raise ValueError(f"Unknown profile type or parallel strategy: {self.profile_type} {self.parallel_strategy}")
    
    def get_task_description(self) -> str:
        """Get the task description"""
        if self.profile_type == "allreduce" and self.parallel_strategy == "dp":
            return "Unconsecutive AllReduce Bandwidth (DP/SDP)"
        elif self.profile_type == "p2p" and self.parallel_strategy == "pp":
            return "P2P Bandwidth (PP)"
        elif self.profile_type == "allreduce" and self.parallel_strategy == "tp":
            return "Consecutive AllReduce (TP)"
        elif self.profile_type == "all2all" and self.parallel_strategy == "sp":
            return "Consecutive All2All (SP)"
        else:
            return f"{self.profile_type} {self.parallel_strategy} Bandwidth"

    def get_args_for_key(self, key: Optional[str] = None) -> Any:
        """Get arguments for bandwidth profiling"""
        if key is None:
            raise ValueError("key is required for bandwidth profiling")
        
        args = HardwareProfileArgs()
        size = int(key.split("_")[-1])  # Extract size from key (e.g., "allreduce_size_8" -> 8)
        if self.profile_type == "allreduce" and self.parallel_strategy == "dp":
            args.global_tp_deg = size
            args.global_tp_consec = 0
            args.pp_deg = 1
            args.local_batch_size = 128
        elif self.profile_type == "allreduce" and self.parallel_strategy == "tp":
            args.global_tp_deg = size
            args.global_tp_consec = 1
            args.pp_deg = 1
            args.local_batch_size = 128
        elif self.profile_type == "p2p" and self.parallel_strategy == "pp":
            args.global_tp_deg = 1
            args.global_tp_consec = 1
            args.pp_deg = size
            args.local_batch_size = 256
        elif self.profile_type == "all2all" and self.parallel_strategy == "sp":
            args.global_tp_deg = size
            args.global_tp_consec = 1
            args.pp_deg = 1
            args.local_batch_size = 128
        args.nproc_per_node = self.num_gpus_per_node
        args.key = key
        return args
