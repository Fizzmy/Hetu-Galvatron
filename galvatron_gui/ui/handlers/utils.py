"""Handler utility functions"""

import json
from pathlib import Path
from typing import Dict


def get_model_config_params_from_name(model_size: str, model_type: str) -> Dict:
    """Get model configuration parameters from galvatron_gui/data/model_configs
    
    Args:
        model_size: Model size (e.g., '7b')
        model_type: Model type (e.g., 'llama')
    
    Returns:
        Dictionary of model configuration parameters with unified key names
        (uses COMMON_PARAMS standard keys: hidden_size, num_hidden_layers, etc.)
    """
    try:
        # Get project root (galvatron_gui directory)
        project_root = Path(__file__).parent.parent.parent
        model_configs_dir = project_root / "data" / "model_configs"
        # {model_type}-{model_size}.json
        # e.g., llama-7b.json
        config_filename = f"{model_type}-{model_size}.json"
        config_file = model_configs_dir / config_filename
        
        if not config_file.exists():
            return {}
        
        # Load raw config (may use different key names)
        with open(config_file, 'r') as f:
            raw_config = json.load(f)
        
        # Map to unified format using standard COMMON_PARAMS keys
        # This allows config files to use different key names (e.g., 'dim' instead of 'hidden_size')
        key_mapping = {
            'hidden_size': ['hidden_size', 'dim', 'd_model'],
            'num_hidden_layers': ['num_hidden_layers', 'n_layers', 'num_layers'],
            'num_attention_heads': ['num_attention_heads', 'n_heads', 'num_heads'],
            'num_key_value_heads': ['num_key_value_heads', 'num_kv_heads', 'n_kv_heads'],
            'intermediate_size': ['intermediate_size', 'ffn_dim', 'd_ff'],
            'max_position_embeddings': ['max_position_embeddings', 'n_positions'],
            'vocab_size': ['vocab_size'],
        }
        
        # Convert to unified format
        unified_config = {}
        for standard_key, possible_keys in key_mapping.items():
            value = None
            for key in possible_keys:
                if key in raw_config:
                    value = raw_config[key]
                    break
            if value is not None:
                unified_config[standard_key] = value
        
        if 'num_key_value_heads' not in unified_config and 'num_attention_heads' in unified_config:
            unified_config['num_key_value_heads'] = unified_config['num_attention_heads']
        
        return unified_config
    except Exception as e:
        import traceback
        print(f"Error loading model config: {e}\n{traceback.format_exc()}")
        return {}
