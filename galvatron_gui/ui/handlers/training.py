"""Training event handlers"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io

import gradio as gr

from ..state import state
from profiling.arguments import TrainingArgs, ModelConfigs

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


def load_available_strategies(model_type: str, model_size: str, cluster_config: str):
    """Load available parallelism strategies for the selected model and cluster
    
    Args:
        model_type: Model type (e.g., 'llama')
        model_size: Model size (e.g., '7b')
        cluster_config: Cluster configuration string (e.g., '1_8' for 1 node, 8 GPUs)
    
    Returns:
        Tuple of (status_display, strategy_accordion_update, strategy_dropdown_update,
                 training_params_accordion_update, training_action_accordion_update)
    """
    if not model_type or not model_size:
        return (
            "❌ Please select both model type and size.",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    if not cluster_config:
        return (
            "❌ Please select cluster configuration.",
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    try:
        # Parse cluster config
        num_nodes, num_gpus_per_node = map(int, cluster_config.split('_'))
        
        model_identifier = f"{model_type}-{model_size}"
        
        # Search for strategy configs in galvatron_gui/data/search_logs/results
        # Match files that contain model_identifier
        results_dir = PROJECT_ROOT / "data" / "search_logs" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all galvatron_config_*.json files that contain model_identifier
        all_strategy_files = []
        for strategy_file in results_dir.glob("galvatron_config_*.json"):
            if model_identifier in strategy_file.name:
                # Also check if it matches the cluster config
                if f"{num_nodes}nodes_{num_gpus_per_node}gpus_per_node" in strategy_file.name:
                    all_strategy_files.append(strategy_file)
        
        lines = [f"### Available Strategies for {model_identifier}\n\n"]
        lines.append(f"**Cluster Configuration:** {num_nodes} Nodes × {num_gpus_per_node} GPUs/Node\n\n")
        
        if all_strategy_files:
            lines.append(f"**Found {len(all_strategy_files)} strategy configuration(s)**\n\n")
            
            # Create dropdown choices
            strategy_choices = []
            for f in all_strategy_files:
                # Display name: just the filename
                display_name = f.name
                # Value: full path
                strategy_choices.append((display_name, str(f)))
            
            lines.append("---\n\n")
            lines.append("✅ **Ready to train!** Select a strategy configuration below.\n")
            
            return (
                "".join(lines),
                gr.update(visible=True),
                gr.update(choices=strategy_choices, value=strategy_choices[0][1] if strategy_choices else None),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            lines.append("---\n\n")
            lines.append(f"❌ **No strategy configurations found.**\n\n")
            lines.append(f"Please run strategy search first to generate configurations.\n\n")
            return (
                "".join(lines),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                gr.update(visible=False),
                gr.update(visible=False),
            )
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error loading strategies: {str(e)}\n\n{traceback.format_exc()}"
        return (
            error_msg,
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def load_gui_config():
    """Load GUI configuration from gui_config.json
    
    Returns:
        Dict with 'datasets', 'tokenizers' lists, and 'ray_env_vars' dict
    """
    config_path = PROJECT_ROOT / "data" / "gui_config.json"
    if not config_path.exists():
        return {"datasets": [], "tokenizers": [], "ray_env_vars": {}}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return {
            "datasets": config.get("datasets", []),
            "tokenizers": config.get("tokenizers", []),
            "ray_env_vars": config.get("ray_env_vars", {})
        }
    except Exception:
        return {"datasets": [], "tokenizers": [], "ray_env_vars": {}}


def scan_directory_for_paths(root_path: str, path_type: str = "dataset"):
    """Scan configured directories for available paths
    
    For datasets: Find directories containing xxx.bin and xxx.idx files,
                  return root_dir/xxx as candidate paths
    For tokenizers: Find directories containing tokenizer.model file,
                    return the directory path
    
    Args:
        root_path: Root directory path to scan (if provided, use this; otherwise use config)
        path_type: Type of path to scan - "dataset" or "tokenizer"
    
    Returns:
        Dropdown update with available paths
    """
    config = load_gui_config()
    
    # If root_path is provided, use it; otherwise use configured directories
    if root_path and root_path.strip():
        search_dirs = [root_path.strip()]
    else:
        # Use configured directories based on path_type
        if path_type == "dataset":
            search_dirs = config.get("datasets", [])
        else:
            search_dirs = config.get("tokenizers", [])
    
    all_paths = []
    
    if path_type == "dataset":
        # For datasets: find xxx.bin files and check for corresponding xxx.idx
        for search_dir in search_dirs:
            if not search_dir:
                continue
            
            root = Path(search_dir)
            if not root.exists() or not root.is_dir():
                continue
            
            try:
                # Recursively search for .bin files
                for bin_file in root.rglob("*.bin"):
                    # Get the base name without extension
                    base_name = bin_file.stem
                    # Check if corresponding .idx file exists in the same directory
                    idx_file = bin_file.parent / f"{base_name}.idx"
                    if idx_file.exists():
                        # Return the directory containing the .bin file + base_name
                        # This preserves the full recursive path structure
                        dataset_path = str(bin_file.parent / base_name)
                        all_paths.append(dataset_path)
            except (PermissionError, OSError):
                continue
    else:
        # For tokenizers: find directories containing tokenizer.model
        for search_dir in search_dirs:
            if not search_dir:
                continue
            
            root = Path(search_dir)
            if not root.exists():
                continue
            
            try:
                # If it's a file, get its parent directory
                if root.is_file():
                    root = root.parent
                
                # Search for tokenizer.model file
                if root.is_dir():
                    # Check current directory
                    tokenizer_file = root / "tokenizer.model"
                    if tokenizer_file.exists():
                        all_paths.append(str(root))
                    
                    # Also check subdirectories
                    for item in root.rglob("tokenizer.model"):
                        tokenizer_dir = item.parent
                        all_paths.append(str(tokenizer_dir))
            except (PermissionError, OSError):
                continue
    
    # Remove duplicates and sort
    all_paths = sorted(list(set(all_paths)))
    return gr.update(choices=all_paths, value=all_paths[0] if all_paths else None)


def display_strategy_details(strategy_path: str):
    """Display details of the selected strategy
    
    Args:
        strategy_path: Path to the strategy configuration file
    
    Returns:
        Markdown string with strategy details
    """
    if not strategy_path:
        return ""
    
    try:
        with open(strategy_path, 'r') as f:
            config = json.load(f)
        
        lines = ["### 📋 Strategy Configuration Details\n\n"]
        lines.append("```json\n")
        lines.append(json.dumps(config, indent=2))
        lines.append("\n```\n\n")
        
        # Parse and display key information
        lines.append("#### Key Parameters:\n\n")
        
        if "pp_deg" in config:
            lines.append(f"- **Pipeline Parallel Degree**: {config['pp_deg']}\n")
        
        if "global_bsz" in config:
            lines.append(f"- **Global Batch Size**: {config['global_bsz']}\n")
        
        if "chunks" in config:
            lines.append(f"- **Pipeline Chunks**: {config['chunks']}\n")
        
        if "pipeline_type" in config:
            lines.append(f"- **Pipeline Type**: {config['pipeline_type']}\n")
        
        if "default_dp_type" in config:
            lines.append(f"- **Default DP Type**: {config['default_dp_type']}\n")
        
        if "vtp" in config:
            lines.append(f"- **Vocabulary Tensor Parallel**: {config['vtp']}\n")
        
        return "".join(lines)
    
    except Exception as e:
        import traceback
        return f"❌ Error reading strategy file: {str(e)}\n\n{traceback.format_exc()}"


def parse_training_metrics_from_logs(log_file: Path, last_rank: Optional[int] = None) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Parse loss and time values from log file
    
    Args:
        log_file: Path to log file
        last_rank: Rank number to extract logs from (optional, typically num_gpus - 1)
    
    Returns:
        Tuple of (losses, times) where each is a list of (iteration, value) tuples
    """
    losses = []
    times = []
    if not log_file.exists():
        return losses, times
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Pattern to match: | Iteration:     17 | ... | Elapsed time per iteration (ms): 11082.1 | ... | Loss: 1.052331e+01 | ...
        # Extract iteration, time, and loss in one pass
        pattern = re.compile(
            r'\|\s+Iteration:\s+(\d+)\s+\|.*?Elapsed time per iteration \(ms\):\s+([\d.eE+-]+)\s+\|.*?\|\s+Loss:\s+([\d.eE+-]+)\s+\|'
        )
        
        for line in lines:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                time_ms = float(match.group(2))
                loss = float(match.group(3))
                losses.append((iteration, loss))
                times.append((iteration, time_ms))
        
        # Sort by iteration
        losses.sort(key=lambda x: x[0])
        times.sort(key=lambda x: x[0])
        return losses, times
    
    except Exception as e:
        print(f"Error parsing log file {log_file}: {e}")
        return losses, times


def get_training_plots(model_type: str, model_size: str) -> Tuple[Optional[str], Optional[str]]:
    """Get loss and time plots for training task using max rank directory
    
    Args:
        model_type: Model type
        model_size: Model size
    
    Returns:
        Tuple of (loss_plot_path, time_plot_path) as file paths, or (None, None)
    """
    if not model_type or not model_size:
        return None, None
    
    try:
        model_identifier = f"{model_type}-{model_size}"
        training_logs_dir = PROJECT_ROOT / "data" / "training_logs"
        
        if not training_logs_dir.exists():
            return None, None
        
        # Find log directory (without timestamp now)
        log_dir = training_logs_dir / f"training_{model_identifier}"
        if not log_dir.exists():
            return None, None
        
        # Log files are now directly in log_dir with format: training_rank_X.log
        # Find all training_rank_*.log files
        log_files = list(log_dir.glob("training_rank_*.log"))
        if not log_files:
            return None, None
        
        # Find the maximum rank number
        max_rank = -1
        latest_log = None
        for log_file in log_files:
            try:
                # Extract rank number from filename: training_rank_X.log
                match = re.search(r'training_rank_(\d+)\.log', log_file.name)
                if match:
                    rank_num = int(match.group(1))
                    if rank_num > max_rank:
                        max_rank = rank_num
                        latest_log = log_file
            except (ValueError, AttributeError):
                continue
        
        if latest_log is None:
            return None, None
        
        # Try to get task info to determine last rank
        last_rank = None
        if state.ray_task_manager:
            task_id = f"training_{model_identifier}"
            task_status = state.ray_task_manager.get_task_status(task_id)
            if task_status and 'config' in task_status:
                config = task_status['config']
                if 'args' in config:
                    args = config['args']
                    if hasattr(args, 'num_nodes') and hasattr(args, 'num_gpus_per_node'):
                        total_gpus = args.num_nodes * args.num_gpus_per_node
                        last_rank = total_gpus - 1
        
        # Parse loss and time values from log
        losses, times = parse_training_metrics_from_logs(latest_log, last_rank)
        
        loss_plot_path = None
        time_plot_path = None
        
        # Create loss plot and save to file
        if losses:
            iterations = [x[0] for x in losses]
            loss_values = [x[1] for x in losses]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, loss_values, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Loss - {model_identifier}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to PNG file in log directory
            loss_plot_path = log_dir / "training_loss_plot.png"
            plt.savefig(loss_plot_path, format='png', dpi=100)
            plt.close()
            loss_plot_path = str(loss_plot_path)
        
        # Create time plot and save to file
        if times:
            # Only show last 10 iterations
            recent_times = times[-10:] if len(times) > 10 else times
            iterations = [x[0] for x in recent_times]
            time_values = [x[1] for x in recent_times]
            
            # Calculate average time
            avg_time = sum(time_values) / len(time_values) if time_values else 0
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, time_values, marker='s', linestyle='-', linewidth=2, markersize=4, color='green')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Time per Iteration (ms)', fontsize=12)
            plt.title(f'Training Time per Iteration - {model_identifier} (Last 10 iterations)', fontsize=14, fontweight='bold')
            
            # Add average time as text annotation
            plt.text(0.02, 0.98, f'Average Time: {avg_time:.2f} ms', 
                    transform=plt.gca().transAxes, 
                    fontsize=12, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to PNG file in log directory
            time_plot_path = log_dir / "training_time_plot.png"
            plt.savefig(time_plot_path, format='png', dpi=100)
            plt.close()
            time_plot_path = str(time_plot_path)
        
        return loss_plot_path, time_plot_path
    
    except Exception as e:
        import traceback
        print(f"Error creating plots: {e}\n{traceback.format_exc()}")
        return None, None


def refresh_training_status(model_type: str, model_size: str):
    """Refresh training status and return status with loss and time plots
    
    Args:
        model_type: Model type
        model_size: Model size
    
    Returns:
        Tuple of (status_display, loss_plot_image, time_plot_image)
    """
    if not model_type or not model_size:
        return "Please select a model first", None
    
    try:
        model_identifier = f"{model_type}-{model_size}"
        
        # Check if there are any training tasks
        training_logs_dir = PROJECT_ROOT / "data" / "training_logs"
        if not training_logs_dir.exists():
            return "No training tasks found.", None
        
        # Find log directory for this model
        log_dir = training_logs_dir / f"training_{model_identifier}"
        
        if not log_dir.exists():
            return "No training tasks found for this model.", None
        
        lines = [f"### Training Tasks for {model_identifier}\n\n"]
        lines.append(f"**Log Directory:** `{log_dir}`\n\n")
        
        # Find rank directories
        rank_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("rank_")]
        
        if rank_dirs:
            # Extract and sort rank numbers
            rank_info = []
            for rank_dir in rank_dirs:
                try:
                    rank_num = int(rank_dir.name.split("_")[1])
                    rank_info.append((rank_num, rank_dir))
                except (ValueError, IndexError):
                    continue
            
            if rank_info:
                rank_info.sort(key=lambda x: x[0])
                max_rank = rank_info[-1][0]
                lines.append(f"**Available Ranks:** {', '.join([f'rank_{r[0]}' for r in rank_info])}\n\n")
                lines.append(f"**Using Max Rank:** `rank_{max_rank}` (for loss plot)\n\n")
            else:
                lines.append("**No valid rank directories found.**\n\n")
        else:
            # Fallback: look for log files
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                lines.append(f"**Total Log Files:** {len(log_files)}\n\n")
                for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    lines.append(f"- `{log_file.name}`\n")
                if len(log_files) > 5:
                    lines.append(f"\n*... and {len(log_files) - 5} more*\n")
            else:
                lines.append("**No log files found.**\n\n")
        
        status_text = "".join(lines)
        
        # Get loss and time plots
        loss_plot, time_plot = get_training_plots(model_type, model_size)
        
        return status_text, loss_plot, time_plot
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error_msg, None


def submit_training(
    model_type: str, model_size: str, cluster_config: str,
    galvatron_config_path: str,
    train_iters: int,
    lr: float, min_lr: float, lr_warmup_fraction: float,
    lr_decay_style: str, adam_weight_decay: float,
    adam_beta1: float, adam_beta2: float, adam_eps: float,
    async_grad_reduce: bool,
    seq_length: int,
    data_path: str, split: str,
    tokenizer_model: str,
    enable_load_checkpoint: bool, load: str,
    enable_save_checkpoint: bool, save: str, save_interval: int,
):
    """Submit distributed training task
    
    Returns:
        Tuple of (output_message, status_display)
    """
    if not model_type or not model_size:
        return "❌ Please select a model first.", ""
    
    if not cluster_config:
        return "❌ Please select cluster configuration.", ""
    
    if not galvatron_config_path:
        return "❌ Please select a parallelism strategy.", ""
    
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
        
        # Use paths directly from dropdown (they are already full paths from config)
        # Convert data_path to list format for --data-path (nargs='*')
        # Format: single prefix -> ["/path/to/data"]
        if data_path:
            # For --data-path with nargs='*', it expects a list
            # Single prefix format: just pass as a list with one element
            data_path_list = [data_path] if isinstance(data_path, str) else data_path
        else:
            data_path_list = None
        tokenizer_model_str = tokenizer_model if tokenizer_model else None
        
        load_str = load if (enable_load_checkpoint and load) else None
        save_str = save if (enable_save_checkpoint and save) else None
        
        # Create TrainingArgs with all parameters
        training_args = TrainingArgs(
            model_type=model_type,
            model_size=model_size,
            model_identifier=model_identifier,
            galvatron_config_path=galvatron_config_path,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            train_iters=int(train_iters),
            lr=float(lr),
            min_lr=float(min_lr),
            lr_warmup_fraction=float(lr_warmup_fraction),
            lr_decay_style=lr_decay_style,
            adam_weight_decay=float(adam_weight_decay),
            adam_beta1=float(adam_beta1),
            adam_beta2=float(adam_beta2),
            adam_eps=float(adam_eps),
            async_grad_reduce=async_grad_reduce,
            seq_length=int(seq_length) if seq_length else 4096,
            data_path=data_path_list,  # List format for --data-path (nargs='*')
            split=split if split else "949,50,1",
            tokenizer_type="HuggingFaceTokenizer",  # Use default value
            tokenizer_model=tokenizer_model_str,
            load=load_str,
            save=save_str,
            save_interval=int(save_interval) if enable_save_checkpoint else 100,
        )
        
        # Get or create training manager from state
        training_manager = state.get_or_create_training_manager(
            training_args=training_args,
            model_configs=model_configs,
        )
        
        # Submit training task using BaseProfileManager's submit_task method
        # This will use Ray placement groups and support multi-GPU parallel execution
        result = training_manager.submit_task()
        
        # Update status and get loss and time plots
        status, loss_plot, time_plot = refresh_training_status(model_type, model_size)
        
        return result, status
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error submitting training: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""
