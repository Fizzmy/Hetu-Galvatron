"""Model Profiling event handlers"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from ..state import state
from .utils import get_model_config_params_from_name
from profiling.arguments import ModelProfileConfigs, ModelConfigs

# Common model parameters to display as editable inputs
# Core parameters (used by GPT, LLaMA, BERT, etc.)
COMMON_PARAMS = [
    'hidden_size', 'num_hidden_layers', 'num_attention_heads',
    'num_key_value_heads', 'intermediate_size',
    'max_position_embeddings', 'vocab_size'
]

def refresh_model_config_cache() -> Dict[str, List[str]]:
    """Refresh model configuration cache from file system
    
    Returns:
        Dictionary mapping model types to their available sizes
    """
    try:
        # Get project root (galvatron_gui directory)
        project_root = Path(__file__).parent.parent.parent
        model_configs_dir = project_root / "data" / "model_configs"
        
        if not model_configs_dir.exists():
            state.model_config_cache = {}
            return state.model_config_cache
        
        # Clear existing cache
        state.model_config_cache = {}
        
        # Parse all config files
        # Format: {model_type}-{model_size}.json (e.g., llama-7b.json)
        for config_file in model_configs_dir.glob("*.json"):
            filename = config_file.stem  # e.g., "llama-7b"
            
            # Try hyphen format: {model_type}-{model_size}
            if "-" in filename:
                parts = filename.split("-", 1)
                if len(parts) >= 2:
                    model_type = parts[0]  # e.g., "llama"
                    model_size = parts[1]  # e.g., "7b"
                    
                    if model_type not in state.model_config_cache:
                        state.model_config_cache[model_type] = []
                    
                    if model_size not in state.model_config_cache[model_type]:
                        state.model_config_cache[model_type].append(model_size)
        
        # Sort sizes for each model type
        for model_type in state.model_config_cache:
            state.model_config_cache[model_type].sort()
        
        return state.model_config_cache
    except Exception as e:
        import traceback
        print(f"Error refreshing model config cache: {e}\n{traceback.format_exc()}")
        state.model_config_cache = {}
        return state.model_config_cache


def get_model_config_cache() -> Dict[str, List[str]]:
    """Get model configuration cache (refresh if empty)
    
    Returns:
        Dictionary mapping model types to their available sizes
    """
    if not state.model_config_cache:
        refresh_model_config_cache()
    return state.model_config_cache


def get_available_model_types() -> List[str]:
    """Get list of available model types from cache
    
    Returns:
        Sorted list of model type names
    """
    cache = get_model_config_cache()
    return sorted(list(cache.keys()))


def update_model_sizes(model_type: str):
    """Update model size dropdown based on selected model type
    
    Args:
        model_type: Selected model type (e.g., 'llama')
    
    Returns:
        Gradio update for model_size_dropdown
    """
    import gradio as gr
    
    if not model_type:
        return gr.update(choices=[], visible=True)
    
    try:
        # Get available sizes for this model type from cache
        cache = get_model_config_cache()
        sizes = cache.get(model_type, [])
        
        if sizes:
            return gr.update(choices=sizes, value=None, visible=True)
        else:
            return gr.update(choices=[], visible=True)
    except Exception as e:
        import traceback
        print(f"Error getting model sizes: {e}\n{traceback.format_exc()}")
        return gr.update(choices=[], visible=True)


def update_profile_mode_choices(profile_type: str):
    """Update profile mode dropdown choices based on selected profile type
    
    Args:
        profile_type: Selected profile type ('computation' or 'memory')
    
    Returns:
        Gradio update for profile_mode_dropdown
    """
    import gradio as gr
    
    if profile_type == "memory":
        # Memory only supports static mode
        return gr.update(choices=[("Static", "static")], value="static", visible=True)
    elif profile_type == "computation":
        # Computation supports all modes
        return gr.update(
            choices=[
                ("Static", "static"),
                ("Batch", "batch"),
                ("Sequence", "sequence")
            ],
            value="static",
            visible=True
        )


def refresh_model_list():
    """Refresh model list and return updated model types
    
    Returns:
        Gradio update for model_type_dropdown with refreshed choices
    """
    import gradio as gr
    
    try:
        # Refresh cache from file system
        refresh_model_config_cache()
        
        # Get updated model types
        model_types = get_available_model_types()
        
        return gr.update(choices=model_types, value=None)
    except Exception as e:
        import traceback
        print(f"Error refreshing model list: {e}\n{traceback.format_exc()}")
        return gr.update(choices=[])


def load_model_config(model_type: str, model_size: str):
    """Load model configuration and return editable parameters
    
    Args:
        model_type: Model type (e.g., 'llama', 'gpt')
        model_size: Model size (e.g., '7b', '13b')
    
    Returns:
        Tuple of gradio updates for UI components and parameter values
    """
    import gradio as gr
    
    if not model_type:
        empty_updates = _get_empty_param_updates()
        return (
            "Please select a model type",
            # Computation profile UI
            gr.update(visible=False),  # comp_profile_mode_dropdown
            gr.update(visible=False),  # comp_params_accordion
            gr.update(visible=False),  # comp_static_params
            gr.update(visible=False),  # comp_batch_params
            gr.update(visible=False),  # comp_sequence_params
            gr.update(visible=False),  # comp_load_config_btn
            # Memory profile UI
            gr.update(visible=False),  # mem_profile_mode_dropdown
            gr.update(visible=False),  # mem_params_accordion
            gr.update(visible=False),  # mem_layernum_row
            gr.update(visible=False),  # mem_load_config_btn
        ) + empty_updates
    
    if not model_size:
        empty_updates = _get_empty_param_updates()
        return (
            "Please select a model size",
            # Computation profile UI
            gr.update(visible=False),  # comp_profile_mode_dropdown
            gr.update(visible=False),  # comp_params_accordion
            gr.update(visible=False),  # comp_static_params
            gr.update(visible=False),  # comp_batch_params
            gr.update(visible=False),  # comp_sequence_params
            gr.update(visible=False),  # comp_load_config_btn
            # Memory profile UI
            gr.update(visible=False),  # mem_profile_mode_dropdown
            gr.update(visible=False),  # mem_params_accordion
            gr.update(visible=False),  # mem_layernum_row
            gr.update(visible=False),  # mem_load_config_btn
        ) + empty_updates
    
    try:
        # Load model configuration parameters
        model_config = get_model_config_params_from_name(model_size, model_type)
        
        # Get computation and memory profile status (default to static mode)
        comp_status = refresh_profile_status(model_type, model_size, "computation", "static")
        mem_status = refresh_profile_status(model_type, model_size, "memory", "static")
        
        # Return parameter values for editable inputs
        param_updates = _get_param_updates(model_config)
        
        return (
            f"✅ Loaded configuration: **{model_size}**\n\nModel type: {model_type}",
            # Computation profile UI
            gr.update(visible=True),  # comp_profile_mode_dropdown
            gr.update(visible=True),  # comp_params_accordion
            gr.update(visible=True),  # comp_static_params
            gr.update(visible=False),  # comp_batch_params
            gr.update(visible=False),  # comp_sequence_params
            gr.update(visible=True),  # comp_load_config_btn
            # Memory profile UI
            gr.update(visible=True),  # mem_profile_mode_dropdown
            gr.update(visible=True),  # mem_params_accordion
            gr.update(visible=True),  # mem_layernum_row
            gr.update(visible=True),  # mem_load_config_btn
        ) + param_updates
    except Exception as e:
        import traceback
        error_msg = f"Error loading model configuration: {str(e)}\n\n{traceback.format_exc()}"
        empty_updates = _get_empty_param_updates()
        return (
            error_msg,
            # Computation profile UI
            gr.update(visible=False),  # comp_profile_mode_dropdown
            gr.update(visible=False),  # comp_params_accordion
            gr.update(visible=False),  # comp_static_params
            gr.update(visible=False),  # comp_batch_params
            gr.update(visible=False),  # comp_sequence_params
            gr.update(visible=False),  # comp_load_config_btn
            # Memory profile UI
            gr.update(visible=False),  # mem_profile_mode_dropdown
            gr.update(visible=False),  # mem_params_accordion
            gr.update(visible=False),  # mem_layernum_row
            gr.update(visible=False),  # mem_load_config_btn
        ) + empty_updates


def _get_param_updates(config: Dict) -> Tuple:
    """Get parameter updates for editable inputs
    
    Args:
        config: Model configuration dictionary with unified key names
                (should use COMMON_PARAMS standard keys from get_model_config_params_from_name)
    
    Returns:
        Tuple of parameter values for all common parameters (in COMMON_PARAMS order)
    """
    # Config is already in unified format (from get_model_config_params_from_name)
    # Just extract values in COMMON_PARAMS order
    updates = []
    for param in COMMON_PARAMS:
        value = config.get(param)  # Direct lookup since keys are unified
        # Convert to appropriate type
        if value is None:
            updates.append(None)
        elif isinstance(value, (int, float)):
            updates.append(value)
        else:
            updates.append(str(value))
    
    # Ensure we return exactly len(COMMON_PARAMS) values
    assert len(updates) == len(COMMON_PARAMS), f"Expected {len(COMMON_PARAMS)} values, got {len(updates)}"
    return tuple(updates)


def _get_empty_param_updates() -> Tuple:
    """Get empty parameter updates (all None)"""
    return tuple([None] * len(COMMON_PARAMS))


def _get_profile_status(model_type: str, model_size: str, profile_type: str, profile_mode: str = "static") -> str:
    """Get profile status display with detailed results
    
    Args:
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        profile_type: Profile type ('computation' or 'memory')
        profile_mode: Profile mode (e.g., 'static', 'batch', 'sequence')
    
    Returns:
        Formatted markdown string with status and detailed results
    """
    lines = [f"### {profile_type.capitalize()} Profile Status ({profile_mode.capitalize()})\n\n"]
    
    model_identifier = f"{model_type}-{model_size}"
    key = (model_identifier, profile_type, profile_mode)
    
    # Check if manager exists
    if key not in state.model_profile_managers:
        lines.append("**Status:** ⚠️ Config not loaded\n")
        lines.append("  - Please click 'Load Config' to create profile configuration\n")
        return "".join(lines)
    
    # Check profile for the specified type and mode
    manager = state.model_profile_managers[key]
    manager._load_data()  # Ensure data is up to date
    completed = manager.is_completed()
    status = "✅" if completed else "❌"
    lines.append(f"**Status:** {status}\n\n")
    
    # Show existing/profiled keys
    profiled_keys = [key for key in manager.need_to_profile if key in manager.data]
    if profiled_keys:
        lines.append(f"**Profiled Keys:** {len(profiled_keys)}/{len(manager.need_to_profile)}\n\n")
        if len(profiled_keys) <= 10:
            for key in profiled_keys:
                lines.append(f"- ✅ `{key}`\n")
        else:
            # Show first 10
            for key in profiled_keys[:10]:
                lines.append(f"- ✅ `{key}`\n")
            lines.append(f"\n... and {len(profiled_keys) - 10} more profiled keys\n")
        lines.append("\n")
    
    # Show missing keys
    missing_keys = manager.get_missing_keys()
    if missing_keys:
        lines.append(f"**Missing Keys:** {len(missing_keys)}/{len(manager.need_to_profile)}\n\n")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                lines.append(f"- ❌ `{key}`\n")
        else:
            # Show first 10
            for key in missing_keys[:10]:
                lines.append(f"- ❌ `{key}`\n")
            lines.append(f"\n... and {len(missing_keys) - 10} more missing keys\n")
    elif not manager.data:
        lines.append("**No profile data available**\n")
    
    # If completed, show detailed results
    if completed and manager.data:
        # Check if results are calculated
        calculated_keys = set()
        if profile_type == "computation":
            calculated_keys = {k for k in manager.data.keys() 
                              if k.startswith("layertype_") or k.startswith("layertype_other")}
        elif profile_type == "memory":
            calculated_keys = {k for k in manager.data.keys() 
                              if k.startswith("layertype_") or k.startswith("other_memory_")}
        has_calculated_results = len(calculated_keys) > 0
        
        # Display calculated results if available
        if has_calculated_results:
            lines.append("\n---\n\n")
            lines.append(f"## 📊 Calculated Results\n\n")
            
            if manager.config_file and manager.config_file.exists():
                lines.append(f"**Config File:** `{manager.config_file.name}`\n\n")
            
            # Display computation results
            if profile_type == "computation":
                # Show layertype results
                layertype_items = [(k, v) for k, v in sorted(manager.data.items()) 
                                  if k.startswith("layertype_") and not k.startswith("layertype_other")]
                if layertype_items:
                    lines.append("### Layer Computation Time\n\n")
                    for k, v in layertype_items:
                        lines.append(f"- **{k}**: `{v:.6f}` seconds\n")
                    lines.append("\n")
                
                # Show other computation overhead
                other_items = [(k, v) for k, v in sorted(manager.data.items()) 
                              if k.startswith("layertype_other")]
                if other_items:
                    lines.append("### Other Computation Overhead\n\n")
                    for k, v in other_items:
                        lines.append(f"- **{k}**: `{v:.6f}` seconds\n")
                    lines.append("\n")
            
            # Display memory results
            elif profile_type == "memory":
                # Show layertype results
                if "layertype_0" in manager.data:
                    lines.append("### Layer Memory Information\n\n")
                    v = manager.data["layertype_0"]
                    if isinstance(v, dict):
                        # v is {seq_length: {parameter_size: ..., tp_activation_per_bsz_dict: {...}}}
                        for seq_k, seq_v in sorted(v.items()):
                            lines.append(f"#### Sequence Length: {seq_k}\n\n")
                            if isinstance(seq_v, dict):
                                for param_k, param_v in sorted(seq_v.items()):
                                    if isinstance(param_v, dict):
                                        # This is tp_activation_per_bsz_dict
                                        lines.append(f"- **{param_k.replace('_', ' ').title()}**:\n")
                                        for tp_k, tp_v in sorted(param_v.items()):
                                            lines.append(f"  - **TP={tp_k}**: `{tp_v:.2f}` MB\n")
                                    else:
                                        # This is parameter_size
                                        lines.append(f"- **{param_k.replace('_', ' ').title()}**: `{param_v:.2f}` MB\n")
                            lines.append("\n")
                    lines.append("\n")
                
                # Show other memory costs
                memory_sections = {
                    "other_memory_pp_off": "Other Memory (Pipeline Parallelism Off)",
                    "other_memory_pp_on_first": "Other Memory (Pipeline Parallelism On - First Stage)",
                    "other_memory_pp_on_last": "Other Memory (Pipeline Parallelism On - Last Stage)"
                }
                
                for k, section_name in memory_sections.items():
                    if k in manager.data:
                        lines.append(f"### {section_name}\n\n")
                        v = manager.data[k]
                        if isinstance(v, dict):
                            for seq_k, seq_v in sorted(v.items()):
                                lines.append(f"#### Sequence Length: {seq_k}\n\n")
                                if isinstance(seq_v, dict):
                                    for sub_k, sub_v in sorted(seq_v.items()):
                                        if isinstance(sub_v, dict):
                                            lines.append(f"#### {sub_k.replace('_', ' ').title()}\n\n")
                                            for sub_sub_k, sub_sub_v in sorted(sub_v.items()):
                                                lines.append(f"- **{sub_sub_k}**: `{sub_sub_v:.2f}` MB\n")
                                            lines.append("\n")
                                        else:
                                            lines.append(f"- **{sub_k}**: `{sub_v:.2f}` MB\n")
                                lines.append("\n")
    
    return "".join(lines)


def refresh_profile_status(model_type: str, model_size: str, profile_type: str, profile_mode: str):
    """Refresh profile status
    
    Args:
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        profile_type: Profile type ('computation' or 'memory')
        profile_mode: Profile mode (e.g., 'static', 'batch', 'sequence')
    
    Returns:
        Status display string
    """
    if not model_type or not model_size:
        return "Please select a model type and size first"
    
    if not profile_mode:
        profile_mode = "static"
    
    try:
        model_identifier = f"{model_type}-{model_size}"
        key = (model_identifier, profile_type, profile_mode)
        
        # Check if manager exists
        if key not in state.model_profile_managers:
            return _get_profile_status(model_type, model_size, profile_type, profile_mode)
        
        manager = state.model_profile_managers[key]
        manager._load_data()  # Ensure data is up to date
        
        # Check if all required profile data is available
        if manager.is_completed():
            # Automatically calculate results if not already calculated
            try:
                manager.calculate_and_save_profiling_results()
            except Exception as calc_error:
                import traceback
                print(f"Warning: Error calculating results: {calc_error}\n{traceback.format_exc()}")
        
        return _get_profile_status(model_type, model_size, profile_type, profile_mode)
    except Exception as e:
        import traceback
        error_msg = f"Error refreshing {profile_type} profile status: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg


def load_profile_config(
    profile_type: str,
    model_type: str, model_size: str, profile_mode: str = None,
    static_batch_size=None, static_seq_length=None,
    batch_min_batch_size=None, batch_max_batch_size=None, batch_batch_size_step=None, batch_seq_length=None,
    sequence_min_seq_length=None, sequence_max_seq_length=None, sequence_seq_length_step=None, sequence_batch_size=None,
    layernum_min=None, layernum_max=None,
    hidden_size=None, num_hidden_layers=None, num_attention_heads=None, num_key_value_heads=None,
    intermediate_size=None, max_position_embeddings=None, vocab_size=None
):
    """Load profile configuration and create ModelProfileManager
    
    Args:
        profile_type: 'computation' or 'memory'
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        profile_mode: Profile mode ('static', 'batch', 'sequence') - required for computation, ignored for memory
        ... other parameters for different modes
    
    Returns:
        Tuple of gradio updates: (status, checkbox_update, refresh_btn_update, submit_btn_update, output_msg)
    """
    import gradio as gr
    
    if not model_type or not model_size:
        return (
            "❌ Please select a model type and size first.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )
    
    # For memory, always use static mode
    if profile_type == "memory":
        profile_mode = "static"
    elif profile_type == "computation":
        if not profile_mode:
            return (
                "❌ Please select a profile mode.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
            )
    
    try:
        # Create ModelConfigs from model parameters
        model_args = ModelConfigs(
            hidden_size=int(hidden_size) if hidden_size else None,
            num_hidden_layers=int(num_hidden_layers) if num_hidden_layers else None,
            num_attention_heads=int(num_attention_heads) if num_attention_heads else None,
            num_key_value_heads=int(num_key_value_heads) if num_key_value_heads else (int(num_attention_heads) if num_attention_heads else None),
            intermediate_size=int(intermediate_size) if intermediate_size else None,
            max_position_embeddings=int(max_position_embeddings) if max_position_embeddings else None,
            vocab_size=int(vocab_size) if vocab_size else None,
        )
        
        # Create ModelProfileConfigs based on mode
        if profile_mode == "static":
            if not static_batch_size or not static_seq_length:
                return (
                    f"❌ Please provide batch size and sequence length for static mode.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                )
            profile_args = ModelProfileConfigs(
                layernum_min=int(layernum_min) if layernum_min else 1,
                layernum_max=int(layernum_max) if layernum_max else 2,
                profile_batch_size=int(static_batch_size),
                profile_seq_length=int(static_seq_length),
            )
        elif profile_mode == "batch":
            if not batch_min_batch_size or not batch_max_batch_size or not batch_batch_size_step or not batch_seq_length:
                return (
                    "❌ Please provide all batch mode parameters.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                )
            profile_args = ModelProfileConfigs(
                layernum_min=int(layernum_min) if layernum_min else 1,
                layernum_max=int(layernum_max) if layernum_max else 2,
                profile_min_batch_size=int(batch_min_batch_size),
                profile_max_batch_size=int(batch_max_batch_size),
                profile_batch_size_step=int(batch_batch_size_step),
                profile_seq_length=int(batch_seq_length),
            )
        elif profile_mode == "sequence":
            if not sequence_min_seq_length or not sequence_max_seq_length or not sequence_seq_length_step or not sequence_batch_size:
                return (
                    "❌ Please provide all sequence mode parameters.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                )
            profile_args = ModelProfileConfigs(
                layernum_min=int(layernum_min) if layernum_min else 1,
                layernum_max=int(layernum_max) if layernum_max else 2,
                profile_min_seq_length=int(sequence_min_seq_length),
                profile_max_seq_length=int(sequence_max_seq_length),
                profile_seq_length_step=int(sequence_seq_length_step),
                profile_batch_size=int(sequence_batch_size),
            )
        else:
            return (
                f"❌ Invalid profile mode: {profile_mode}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
            )
        
        # Create ModelProfileManager
        model_identifier = f"{model_type}-{model_size}"
        manager = state.get_or_create_model_profile_manager(
            model_identifier, profile_type, profile_mode,
            profile_args, model_args
        )
        
        # Automatically refresh status to get detailed information
        # This will also trigger automatic calculation if all data is available
        status = refresh_profile_status(model_type, model_size, profile_type, profile_mode)
        
        # Generate success message
        if profile_type == "computation":
            success_msg = f"✅ Computation profile config loaded for {profile_mode} mode"
        else:
            success_msg = "✅ Memory profile config loaded"
        
        return (
            status,
            gr.update(visible=True),  # profile_missing_only_checkbox
            gr.update(visible=True),  # refresh_status_btn
            gr.update(visible=True),  # submit_profile_btn
            success_msg,
        )
    except Exception as e:
        import traceback
        error_msg = f"❌ Error loading {profile_type} profile config: {str(e)}\n\n{traceback.format_exc()}"
        return (
            error_msg,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )


def submit_model_profile(
    profile_type: str,
    model_type: str, model_size: str, profile_mode: str = None,
    static_batch_size=None, static_seq_length=None,
    batch_min_batch_size=None, batch_max_batch_size=None, batch_batch_size_step=None, batch_seq_length=None,
    sequence_min_seq_length=None, sequence_max_seq_length=None, sequence_seq_length_step=None, sequence_batch_size=None,
    layernum_min=None, layernum_max=None,
    hidden_size=None, num_hidden_layers=None, num_attention_heads=None,
    intermediate_size=None, max_position_embeddings=None, vocab_size=None,
    profile_missing_only: bool = False
):
    """Submit model profiling task
    
    Args:
        profile_type: Type of profile ('computation' or 'memory')
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        profile_mode: Profile mode ('static', 'batch', or 'sequence')
        static_batch_size: Batch size for static mode
        static_seq_length: Sequence length for static mode
        batch_min_batch_size: Min batch size for batch mode
        batch_max_batch_size: Max batch size for batch mode
        batch_batch_size_step: Batch size step for batch mode
        batch_seq_length: Sequence length for batch mode
        sequence_min_seq_length: Min sequence length for sequence mode
        sequence_max_seq_length: Max sequence length for sequence mode
        sequence_seq_length_step: Sequence length step for sequence mode
        sequence_batch_size: Batch size for sequence mode
        layernum_min: Minimum layer number
        layernum_max: Maximum layer number
        hidden_size: Model hidden size
        num_hidden_layers: Number of hidden layers
        num_attention_heads: Number of attention heads
        intermediate_size: Intermediate size
        max_position_embeddings: Max position embeddings
        vocab_size: Vocabulary size
        profile_missing_only: Whether to only profile missing data
    
    Returns:
        Status message
    """
    if not model_type:
        return "❌ Please select a model type first."
    
    if not model_size:
        return "❌ Please select a model size first."
    
    if not profile_type:
        return "❌ Please select a profile type."
    
    # For memory, always use static mode
    if profile_type == "memory":
        profile_mode = "static"
    elif profile_type == "computation":
        if not profile_mode:
            return "❌ Please select a profile mode."
    
    # Validate: memory only supports static mode
    if profile_type == "memory" and profile_mode != "static":
        return "❌ Memory profiling only supports static mode."
    
    try:
        # First, ensure the profile config is loaded/updated with current parameters
        # This will create or update the manager with latest parameters
        load_result = load_profile_config(
            profile_type=profile_type,
            model_type=model_type,
            model_size=model_size,
            profile_mode=profile_mode,
            static_batch_size=static_batch_size,
            static_seq_length=static_seq_length,
            batch_min_batch_size=batch_min_batch_size,
            batch_max_batch_size=batch_max_batch_size,
            batch_batch_size_step=batch_batch_size_step,
            batch_seq_length=batch_seq_length,
            sequence_min_seq_length=sequence_min_seq_length,
            sequence_max_seq_length=sequence_max_seq_length,
            sequence_seq_length_step=sequence_seq_length_step,
            sequence_batch_size=sequence_batch_size,
            layernum_min=layernum_min,
            layernum_max=layernum_max,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
        )
        
        # Check if loading config failed
        if "❌" in load_result[-1]:
            return load_result[-1]
        
        # Get the manager and submit task
        model_identifier = f"{model_type}-{model_size}"
        key = (model_identifier, profile_type, profile_mode)
        
        if key not in state.model_profile_managers:
            return "❌ Failed to create profile manager. Please check your parameters."
        
        manager = state.model_profile_managers[key]
        return manager.submit_task(profile_missing_only=profile_missing_only)
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
