"""Search event handlers"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

from ..state import state
from profiling.model import ModelProfileManager
from profiling.arguments import SearchArgs, ModelConfigs

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_model_config_cache() -> Dict[str, List[str]]:
    """Get model configuration cache"""
    from .model_profiling import get_model_config_cache as get_cache
    return get_cache()


def get_available_model_types() -> List[str]:
    """Get list of available model types"""
    cache = get_model_config_cache()
    return sorted(list(cache.keys()))


def refresh_model_list():
    """Refresh model list"""
    from .model_profiling import refresh_model_list as refresh
    return refresh()


def update_model_sizes(model_type: str):
    """Update model size dropdown based on selected model type"""
    from .model_profiling import update_model_sizes as update
    return update(model_type)


def load_profiling_results(model_type: str, model_size: str, cluster_config: str):
    """Load and check profiling results for the selected model
    
    Args:
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        cluster_config: Cluster configuration string (e.g., '1_8' for 1 node, 8 GPUs)
    
    Returns:
        Tuple of (status_display, profiling_config_row_update, comp_profile_dropdown_update, 
                 mem_profile_dropdown_update, search_params_accordion_update, search_action_accordion_update)
    """
    if not model_type or not model_size:
        return (
            "❌ Please select both model type and size.",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    if not cluster_config:
        return (
            "❌ Please select cluster configuration.",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    try:
        # Parse cluster config
        num_nodes, num_gpus_per_node = map(int, cluster_config.split('_'))
        
        model_identifier = f"{model_type}-{model_size}"
        
        # Check profiling results
        lines = [f"### Profiling Results for {model_identifier}\n\n"]
        lines.append(f"**Cluster Configuration:** {num_nodes} Nodes × {num_gpus_per_node} GPUs/Node\n\n")
        
        # Check computation profiling (try different modes)
        comp_results = []
        for mode in ["static", "batch", "sequence", "hybrid"]:
            config_file = PROJECT_ROOT / "data" / "profiling_results" / "model" / f"computation_profiling_{model_identifier}_{mode}.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    # Check if has calculated results
                    has_calculated = any(k.startswith("layertype_") or k.startswith("layertype_other") for k in data.keys())
                    if has_calculated:
                        comp_results.append(mode)
        
        # Check memory profiling (all modes)
        mem_results = []
        for mode in ["static", "batch", "sequence", "hybrid"]:
            mem_config_file = PROJECT_ROOT / "data" / "profiling_results" / "model" / f"memory_profiling_{model_identifier}_{mode}.json"
            if mem_config_file.exists():
                with open(mem_config_file, 'r') as f:
                    data = json.load(f)
                    # Check if has calculated results
                    has_calculated = any(k.startswith("layertype_") or k.startswith("other_memory_") for k in data.keys())
                    if has_calculated:
                        mem_results.append(mode)
        
        # Display status
        if comp_results:
            lines.append(f"**Computation Profiling:** ✅ Available ({', '.join(comp_results)})\n")
        else:
            lines.append(f"**Computation Profiling:** ❌ No calculated results\n")
        
        if mem_results:
            lines.append(f"**Memory Profiling:** ✅ Available ({', '.join(mem_results)})\n")
        else:
            lines.append(f"**Memory Profiling:** ❌ No calculated results\n")
        
        lines.append("\n")
        
        # Check hardware profiling
        from .hardware_profiling import check_bandwidth_profiles_status
        hw_status = check_bandwidth_profiles_status(num_nodes, num_gpus_per_node)
        
        all_hw_ready = all(hw_status.values())
        if all_hw_ready:
            lines.append(f"**Hardware Profiling:** ✅ All bandwidth profiles available\n")
        else:
            lines.append(f"**Hardware Profiling:** ⚠️ Some bandwidth profiles missing\n")
            missing = [k for k, v in hw_status.items() if not v]
            for m in missing:
                lines.append(f"  - ❌ {m}\n")
        
        lines.append("\n")
        
        # Overall status
        can_search = bool(comp_results) and bool(mem_results) and all_hw_ready
        if can_search:
            lines.append("---\n\n")
            lines.append("✅ **Ready to search!** All required profiling data is available.\n")
            lines.append("\nPlease select the profiling configurations you want to use below.\n")
            
            # Create dropdown choices (with capitalized labels)
            comp_choices = [(mode.capitalize(), mode) for mode in comp_results]
            mem_choices = [(mode.capitalize(), mode) for mode in mem_results]
            
            return (
                "".join(lines),
                gr.update(visible=True),
                gr.update(choices=comp_choices, value=comp_results[0] if comp_results else None),
                gr.update(choices=mem_choices, value=mem_results[0] if mem_results else None),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            lines.append("---\n\n")
            lines.append("❌ **Cannot search yet.** Please complete the missing profiling tasks first.\n")
            return (
                "".join(lines),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(visible=False),
                gr.update(visible=False),
            )
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error loading profiling results: {str(e)}\n\n{traceback.format_exc()}"
        return (
            error_msg,
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def submit_search(
    model_type: str, model_size: str, cluster_config: str,
    comp_profile_mode: str, mem_profile_mode: str,
    batch_size: int, settle_chunk: int,
    memory_constraint: int, seq_length: int,
    search_space: str, sp_space: str,
    max_tp_deg: int, max_pp_deg: int,
    disable_dp: bool, disable_tp: bool, disable_pp: bool, disable_sdp: bool,
    disable_ckpt: bool, disable_vtp: bool, disable_tp_consec: bool,
    no_global_memory_buffer: bool, no_async_grad_reduce: bool,
    pipeline_type: str, default_dp_type: str, mixed_precision: str,
    fine_grained_mode: bool, sequence_parallel: bool,
):
    """Submit parallelism strategy search task
    
    Returns:
        Tuple of (output_message, status_display)
    """
    if not model_type or not model_size:
        return "❌ Please select a model first.", ""
    
    if not cluster_config:
        return "❌ Please select cluster configuration.", ""
    
    if not comp_profile_mode or not mem_profile_mode:
        return "❌ Please select profiling configurations.", ""
    
    try:
        # Parse cluster config
        num_nodes, num_gpus_per_node = map(int, cluster_config.split('_'))
        
        model_identifier = f"{model_type}-{model_size}"
        
        # Load model configs from file
        from .utils import get_model_config_params_from_name
        model_config_dict = get_model_config_params_from_name(model_size, model_type)
        model_configs = ModelConfigs(
            hidden_size=model_config_dict.get('hidden_size'),
            num_hidden_layers=model_config_dict.get('num_hidden_layers'),
            num_attention_heads=model_config_dict.get('num_attention_heads'),
            num_key_value_heads=model_config_dict.get('num_key_value_heads'),
            intermediate_size=model_config_dict.get('intermediate_size'),
            max_position_embeddings=model_config_dict.get('max_position_embeddings'),
            vocab_size=model_config_dict.get('vocab_size'),
        )
        
        # Build profiling and hardware config paths based on selected configurations
        # Profiling paths
        gui_model_dir = str(PROJECT_ROOT / "data" / "profiling_results" / "model")
        gui_hardware_dir = str(PROJECT_ROOT / "data" / "profiling_results" / "hardware")
        # Output and log paths
        search_results_dir = PROJECT_ROOT / "data" / "search_logs" / "results"
        search_logs_dir = PROJECT_ROOT / "data" / "search_logs" / "logs"
        search_results_dir.mkdir(parents=True, exist_ok=True)
        search_logs_dir.mkdir(parents=True, exist_ok=True)

        output_config_path = str(search_results_dir)
        log_dir = str(search_logs_dir)
        
        # Create SearchArgs with all parameters
        # For fixed batch size search: min_bsz = max_bsz = batch_size, settle_bsz = batch_size
        search_args = SearchArgs(
            model_type=model_type,
            model_size=model_identifier,
            model_identifier=model_identifier,
            comp_profile_mode=comp_profile_mode,
            mem_profile_mode=mem_profile_mode,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            min_bsz=int(batch_size),
            max_bsz=int(batch_size),
            settle_bsz=int(batch_size),
            settle_chunk=int(settle_chunk),
            memory_constraint=int(memory_constraint),
            seq_length=int(seq_length),
            search_space=search_space,
            sp_space=sp_space,
            max_tp_deg=int(max_tp_deg),
            max_pp_deg=int(max_pp_deg),
            disable_dp=int(disable_dp),
            disable_tp=int(disable_tp),
            disable_pp=int(disable_pp),
            disable_sdp=int(disable_sdp),
            disable_ckpt=int(disable_ckpt),
            disable_vtp=int(disable_vtp),
            disable_tp_consec=int(disable_tp_consec),
            global_memory_buffer=not no_global_memory_buffer,
            async_grad_reduce=not no_async_grad_reduce,
            pipeline_type=pipeline_type,
            default_dp_type=default_dp_type,
            mixed_precision=mixed_precision,
            fine_grained_mode=int(fine_grained_mode),
            sequence_parallel=sequence_parallel,
            # Auto-set paths
            gui_hardware_dir=gui_hardware_dir,
            gui_model_dir=gui_model_dir,
            output_config_path=output_config_path,
            log_dir=log_dir,
            time_profile_mode=comp_profile_mode,
            memory_profile_mode=mem_profile_mode,
        )
        
        # Get or create search manager from state
        search_manager = state.get_or_create_search_manager(
            search_args=search_args,
            model_configs=model_configs,
        )
        
        # Submit search task
        # This will use Ray placement groups and support multi-CPU parallel execution
        result = search_manager.submit_task()
        
        # Update status
        status = refresh_search_status(model_type, model_size)
        
        return result, status
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error submitting search: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


def refresh_search_status(model_type: str, model_size: str):
    """Refresh search status
    
    Args:
        model_type: Model type
        model_size: Model size
    
    Returns:
        Status display string
    """
    if not model_type or not model_size:
        return "Please select a model first"
    
    try:
        model_identifier = f"{model_type}-{model_size}"
        
        # Check if there are any search tasks
        search_logs_dir = PROJECT_ROOT / "data" / "search_logs"
        if not search_logs_dir.exists():
            return "No search tasks found."
        
        # Find logs for this model
        log_files = list(search_logs_dir.glob(f"search_{model_identifier}_*.log"))
        
        if not log_files:
            return "No search tasks found for this model."
        
        lines = [f"### Search Tasks for {model_identifier}\n\n"]
        lines.append(f"**Total Tasks:** {len(log_files)}\n\n")
        
        # Show recent logs (up to 5)
        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            lines.append(f"- `{log_file.name}`\n")
        
        if len(log_files) > 5:
            lines.append(f"\n*... and {len(log_files) - 5} more*\n")
        
        return "".join(lines)
    
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"
