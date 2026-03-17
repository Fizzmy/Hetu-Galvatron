"""Model profiling management"""

import json
import os
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .base import BaseProfileManager
from .arguments import ModelProfileConfigs, ModelConfigs, ModelProfileArgs

# Get project root (galvatron_gui directory)
PROJECT_ROOT = Path(__file__).parent.parent
GALVATRON_ROOT = Path(PROJECT_ROOT.parent) / "galvatron"
MODEL_CONFIGS_DIR = PROJECT_ROOT / "data" / "profiling_results" / "model"
MODEL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

class ModelProfileManager(BaseProfileManager):
    """Manages model profiling tasks"""
    
    def __init__(self, num_gpus_per_node: int, model_name: str, profile_type: str, profile_mode: str, profile_args: ModelProfileConfigs, model_args: ModelConfigs):
        """
        Initialize model profile manager
        
        Args:
            model_name: Name of the model (e.g., 'llama-7b', 'gpt-7b')
            profile_type: Type of profiling ('computation' or 'memory')
            profile_mode: Profile mode ('static', 'batch', or 'sequence')
        """
        self.model_name = model_name
        self.profile_type = profile_type
        self.profile_mode = profile_mode
        self.num_gpus_per_node = num_gpus_per_node
        self.profile_args = profile_args
        self.model_args = model_args
        
        # Create a config file path (even though we load from multiple files)
        # Format: {model_name}_{profile_type}_{profile_mode}.json
        config_file = MODEL_CONFIGS_DIR / f"{profile_type}_profiling_{model_name}_{profile_mode}.json"
        
        super().__init__(config_file)

        self.need_to_profile = []
        if profile_type == "computation":
            self.total_gpus = 1
            if profile_mode == "static":
                self.need_to_profile.append(f"layernum[{profile_args.layernum_min}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
                self.need_to_profile.append(f"layernum[{profile_args.layernum_max}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            elif profile_mode == "batch":
                for batch_size in range(profile_args.profile_min_batch_size, profile_args.profile_max_batch_size + 1, profile_args.profile_batch_size_step):
                    self.need_to_profile.append(f"layernum[{profile_args.layernum_min}]_bsz{batch_size}_seq{profile_args.profile_seq_length}")
                    self.need_to_profile.append(f"layernum[{profile_args.layernum_max}]_bsz{batch_size}_seq{profile_args.profile_seq_length}")
            elif profile_mode == "sequence":
                for seq_length in range(profile_args.profile_min_seq_length, profile_args.profile_max_seq_length + 1, profile_args.profile_seq_length_step):
                    self.need_to_profile.append(f"layernum[{profile_args.layernum_min}]_bsz{profile_args.profile_batch_size}_seq{seq_length}")
                    self.need_to_profile.append(f"layernum[{profile_args.layernum_max}]_bsz{profile_args.profile_batch_size}_seq{seq_length}")
            else:
                raise ValueError(f"Invalid profile mode: {profile_mode}")
        elif profile_type == "memory":
            self.total_gpus = self.num_gpus_per_node
            assert profile_mode == "static", "Memory profiling only supports static mode"
            self.need_to_profile.append(f"pp1_ckpt0_layernum[{profile_args.layernum_min}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            self.need_to_profile.append(f"pp1_ckpt0_layernum[{profile_args.layernum_max}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            self.need_to_profile.append(f"pp1_ckpt1_layernum[{profile_args.layernum_min}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            self.need_to_profile.append(f"pp1_ckpt1_layernum[{profile_args.layernum_max}]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            if self.num_gpus_per_node >= 2:
                self.need_to_profile.append(f"pp2_ckpt0_layernum[2]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            if self.num_gpus_per_node >= 4:
                self.need_to_profile.append(f"pp4_ckpt0_layernum[4]_bsz{profile_args.profile_batch_size}_seq{profile_args.profile_seq_length}")
            
        else:
            raise ValueError(f"Invalid profile type: {profile_type}")

    def get_required_keys(self) -> List[str]:
        """Get required keys for model profiling"""
        return self.need_to_profile

    def get_status(self) -> str:
        """Get status message"""
        if self.is_completed():
            num_keys = len(self.data)
            return f"✅ Completed - {self.model_name} {self.profile_type} {self.profile_mode}: {num_keys} keys profiled"
        else:
            missing_keys = self.get_missing_keys()
            num_missing = len(missing_keys)
            num_total = len(self.need_to_profile)
            return f"❌ Not completed - {num_missing}/{num_total} keys missing"
    
    def get_profile_function(self):
        """Get the Ray remote function for model profiling"""
        # TODO: init more flexable
        from galvatron.models.llama_hf.train_dist_random import train_remote
        return train_remote
    
    def get_task_type(self) -> str:
        """Get the task type identifier"""
        return f"profile_{self.model_name}_{self.profile_type}_{self.profile_mode}"
    
    def get_task_description(self) -> str:
        """Get the task description"""
        return f"{self.model_name} {self.profile_type} {self.profile_mode} Profiling"
    
    def get_args_for_key(self, key: Optional[str] = None) -> Any:
        """Get arguments for model profiling"""
        # For model profiling, we don't use args in the traditional sense
        # The key contains all the information needed
        # Return a simple object with the key
        args = ModelProfileArgs()
        
        for k, v in self.model_args.__dict__.items():
            setattr(args, k, v)
        if self.profile_type == "computation":
            pattern = re.compile(r"layernum\[(\d+)\]_bsz(\d+)_seq(\d+)")
            match = pattern.match(key)
            if match:
                layernum = int(match.group(1))
                bsz = int(match.group(2))
                seq = int(match.group(3))
            else:
                raise ValueError(f"Invalid key: {key}")
            args.profile_mode = self.profile_mode
            args.model_size = self.model_name
            args.global_train_batch_size = bsz
            args.seq_length = seq
            args.save_profiled_memory = 0
            args.profile_forward = 1
            args.sdp = 0
            args.embed_sdp = 0
            args.default_dp_type = "ddp"
            args.num_hidden_layers = layernum
            args.pp_deg = 1
            args.global_tp_deg = 1
            args.global_checkpoint = 0
        elif self.profile_type == "memory":
            pattern = re.compile(r"pp(\d+)_ckpt(\d+)_layernum\[(\d+)\]_bsz(\d+)_seq(\d+)")
            match = pattern.match(key)
            if match:
                pp_deg = int(match.group(1))
                ckpt = int(match.group(2))
                layernum = int(match.group(3))
                bsz = int(match.group(4))
                seq = int(match.group(5))
            else:
                raise ValueError(f"Invalid key: {key}")
            args.profile_mode = self.profile_mode
            args.model_size = self.model_name
            args.global_train_batch_size = bsz
            args.seq_length = seq
            args.save_profiled_memory = 1
            args.profile_forward = 0
            args.sdp = 1
            args.embed_sdp = 1
            args.default_dp_type = "zero3"
            args.num_hidden_layers = layernum
            args.pp_deg = pp_deg
            args.global_tp_deg = 1
            args.global_checkpoint = ckpt
        args.key = key
        return args

    def calculate_and_save_profiling_results(self):
        """Calculate and save profiling results"""
        if self.profile_type == "computation":
            self.calculate_computation_results()
        elif self.profile_type == "memory":
            self.calculate_memory_results()

    def calculate_computation_results(self):
        """Calculate computation results from profiled data
        
        This method:
        1. Reads profiled computation time data
        2. Calculates per-layer computation time
        3. Calculates other computation overhead
        4. Processes results for different batch sizes and sequence lengths
        5. Saves processed results to config file
        """
        self._load_data()
        
        if not self.data:
            return  # No data to process
        
        # Get batch sizes and sequence lengths based on profile mode
        if self.profile_mode == "static":
            batch_sizes = [self.profile_args.profile_batch_size]
            seq_lengths = [self.profile_args.profile_seq_length]
        elif self.profile_mode == "batch":
            batch_sizes = list(range(
                self.profile_args.profile_min_batch_size,
                self.profile_args.profile_max_batch_size + 1,
                self.profile_args.profile_batch_size_step
            ))
            seq_lengths = [self.profile_args.profile_seq_length]
        elif self.profile_mode == "sequence":
            batch_sizes = [self.profile_args.profile_batch_size]
            seq_lengths = list(range(
                self.profile_args.profile_min_seq_length,
                self.profile_args.profile_max_seq_length + 1,
                self.profile_args.profile_seq_length_step
            ))
        else:
            raise ValueError(f"Invalid profile mode: {self.profile_mode}")
        
        layernum_min = self.profile_args.layernum_min
        layernum_max = self.profile_args.layernum_max
        layernum_diff = layernum_max - layernum_min
        
        # Process each batch size and sequence length combination
        for bsz in batch_sizes:
            for seq in seq_lengths:
                # Get base configuration (layernum_min)
                key_base = f"layernum[{layernum_min}]_bsz{bsz}_seq{seq}"
                if key_base not in self.data:
                    raise ValueError(f"Base key {key_base} not found in data")
                
                val_base = self.data[key_base]
                
                # Get max configuration (layernum_max)
                key_max = f"layernum[{layernum_max}]_bsz{bsz}_seq{seq}"
                if key_max not in self.data:
                    raise ValueError(f"Max key {key_max} not found in data")
                
                val_max = self.data[key_max]
                
                # Calculate per-layer computation time
                # avg_time = (max_time - base_time) / batch_size / (layernum_max - layernum_min)
                avg_time = (val_max - val_base) / bsz / layernum_diff
                
                # Save per-layer time
                write_key = f"layertype_0_bsz{bsz}_seq{seq}"
                self.data[write_key] = avg_time
                
                # Calculate other computation overhead
                # other_time = base_time - (layernum_min * avg_time * batch_size) / batch_size
                other_time = val_base - (layernum_min * avg_time * bsz)
                other_time = other_time / bsz if bsz > 0 else 0
                other_time = max(other_time, 0)  # Ensure non-negative
                
                # Save other overhead
                write_key_other = f"layertype_other_bsz{bsz}_seq{seq}"
                self.data[write_key_other] = other_time
        
        # Save processed results
        self._save_data()

    def calculate_memory_results(self):
        """Calculate memory results from profiled data
        
        This method:
        1. Reads profiled memory data
        2. Calculates per-layer parameter and activation memory
        3. Calculates memory overhead for different parallelism strategies
        4. Processes checkpointing memory costs
        5. Saves processed results to config file
        """
        self._load_data()
        
        if not self.data:
            return  # No data to process
        
        # Memory profiling only supports static mode
        assert self.profile_mode == "static", "Memory profiling only supports static mode"
        
        bsz = self.profile_args.profile_batch_size
        seq = self.profile_args.profile_seq_length
        layernum_min = self.profile_args.layernum_min
        layernum_max = self.profile_args.layernum_max
        layernum_diff = layernum_max - layernum_min

        world_size = self.num_gpus_per_node
        
        # Initialize result containers (simplified: single layer type)
        param_per_layer = 0
        act_per_layer_per_sample = {}
        
        # Process pp1_ckpt0 (base configuration for parameter and activation memory)
        key_min_ckpt0 = f"pp1_ckpt0_layernum[{layernum_min}]_bsz{bsz}_seq{seq}"
        key_max_ckpt0 = f"pp1_ckpt0_layernum[{layernum_max}]_bsz{bsz}_seq{seq}"
        
        if key_min_ckpt0 in self.data and key_max_ckpt0 in self.data:
            min_data = self.data[key_min_ckpt0]
            max_data = self.data[key_max_ckpt0]
            
            # Get rank 0 data (assuming TP=1, so all ranks have same model states)
            rank0_ms_min = min_data.get(f"rank0_ms", 0)
            rank0_ms_max = max_data.get(f"rank0_ms", 0)
            
            # Calculate parameter memory per layer
            param_per_layer = ((rank0_ms_max - rank0_ms_min) / layernum_diff) / 4 
            param_per_layer *= world_size
            
            # Calculate activation memory per sample (for TP=1)
            rank0_act_min = min_data.get(f"rank0_act", 0)
            rank0_act_max = max_data.get(f"rank0_act", 0)
            act_per_layer_per_sample[1] = ((rank0_act_max - rank0_act_min) / layernum_diff)
            act_per_layer_per_sample[1] *= world_size / bsz
        
        # Process pp1_ckpt1 (checkpointing activation memory)
        key_min_ckpt1 = f"pp1_ckpt1_layernum[{layernum_min}]_bsz{bsz}_seq{seq}"
        key_max_ckpt1 = f"pp1_ckpt1_layernum[{layernum_max}]_bsz{bsz}_seq{seq}"
        
        if key_min_ckpt1 in self.data and key_max_ckpt1 in self.data:
            min_data = self.data[key_min_ckpt1]
            max_data = self.data[key_max_ckpt1]
            
            rank0_act_min = min_data.get(f"rank0_act", 0)
            rank0_act_max = max_data.get(f"rank0_act", 0)
            act_per_layer_per_sample["checkpoint"] = ((rank0_act_max - rank0_act_min) / layernum_diff)
            act_per_layer_per_sample["checkpoint"] *= world_size / bsz

        # Process other memory costs using loop (pp1, pp2, pp4)
        inf = 1e6
        other_memory_pp_off = {"model_states": {1: inf}, "activation": {1: inf}}
        other_memory_pp_on_first = {"model_states": {1: inf}, "activation": {1: inf}}
        other_memory_pp_on_last = {"model_states": {1: inf}, "activation": {1: inf}}
        
        tp_deg = 1
        pp_deg = 1
        while pp_deg <= world_size:
            # Determine layernum and key
            if pp_deg == 1:
                layernum = layernum_min
                key = key_min_ckpt0
            else:
                layernum = pp_deg
                key = f"pp{pp_deg}_ckpt0_layernum[{layernum}]_bsz{bsz}_seq{seq}"
            
            if key not in self.data:
                pp_deg *= 2
                continue
            
            data = self.data[key]
            
            # Calculate per-layer memory costs
            ms_cost_per_layer = param_per_layer * 4
            act_cost_per_layer = act_per_layer_per_sample.get(1, 0)
            
            # TODO: consider layernum
            layer_ms_first = ms_cost_per_layer
            layer_ms_last = ms_cost_per_layer
            layer_act_first = act_cost_per_layer
            layer_act_last = act_cost_per_layer
            
            # Get memory data for first and last stages
            rank_first = 0
            rank_last = world_size - 1
            
            rank_first_ms = data.get(f"rank{rank_first}_ms", 0)
            rank_last_ms = data.get(f"rank{rank_last}_ms", 0)
            rank_first_act_peak = max(
                data.get(f"rank{rank_first}_act_peak", 0),
                data.get(f"rank{rank_first}_act", 0)
            )
            rank_last_act_peak = max(
                data.get(f"rank{rank_last}_act_peak", 0),
                data.get(f"rank{rank_last}_act", 0)
            )
            
            # Calculate other memory costs (ZeRO-3 adjusted)
            dp_deg = world_size // pp_deg // tp_deg
            other_ms_first = (
                (rank_first_ms - layer_ms_first / dp_deg)
                * (world_size // pp_deg)
                / tp_deg
            )
            other_ms_last = (
                (rank_last_ms - layer_ms_last / dp_deg)
                * (world_size // pp_deg)
                / tp_deg
            )
            
            bsz_per_rank = bsz / (world_size // (pp_deg * tp_deg))
            bsz_per_world_pp = bsz / world_size * pp_deg # TODO: VTP
            other_act_first = (rank_first_act_peak - layer_act_first * bsz_per_rank) / bsz_per_world_pp
            other_act_last = (rank_last_act_peak - layer_act_last * bsz_per_rank) / bsz_per_world_pp
        
            # Ensure non-negative values
            other_ms_first = max(other_ms_first, 0)
            other_ms_last = max(other_ms_last, 0)
            other_act_first = max(other_act_first, 0)
            other_act_last = max(other_act_last, 0)
            
            # Update memory dictionaries
            tp_key = 1
            if pp_deg == 1:
                other_memory_pp_off["model_states"][tp_key] = min(
                    other_memory_pp_off["model_states"][tp_key], other_ms_first
                )
                other_memory_pp_off["activation"][tp_key] = min(
                    other_memory_pp_off["activation"][tp_key], other_act_first
                )
            else:
                other_memory_pp_on_first["model_states"][tp_key] = min(
                    other_memory_pp_on_first["model_states"][tp_key], other_ms_first
                )
                other_memory_pp_on_first["activation"][tp_key] = min(
                    other_memory_pp_on_first["activation"][tp_key], other_act_first
                )
                other_memory_pp_on_last["model_states"][tp_key] = min(
                    other_memory_pp_on_last["model_states"][tp_key], other_ms_last
                )
                other_memory_pp_on_last["activation"][tp_key] = min(
                    other_memory_pp_on_last["activation"][tp_key], other_act_last
                )
            
            pp_deg *= 2
        
        # Store results in simplified format
        # For single layer type, use layertype_0
        seq_str = str(seq)
        if "layertype_0" not in self.data:
            self.data["layertype_0"] = {}
        if "other_memory_pp_off" not in self.data:
            self.data["other_memory_pp_off"] = {}
        if "other_memory_pp_on_first" not in self.data:
            self.data["other_memory_pp_on_first"] = {}
        if "other_memory_pp_on_last" not in self.data:
            self.data["other_memory_pp_on_last"] = {}
        self.data["layertype_0"][seq_str] = {
            "parameter_size": param_per_layer,
            "tp_activation_per_bsz_dict": act_per_layer_per_sample,
        }
        
        # Store other memory costs
        self.data["other_memory_pp_off"][seq_str] = other_memory_pp_off
        self.data["other_memory_pp_on_first"][seq_str] = other_memory_pp_on_first
        self.data["other_memory_pp_on_last"][seq_str] = other_memory_pp_on_last
        
        # Save processed results
        self._save_data()