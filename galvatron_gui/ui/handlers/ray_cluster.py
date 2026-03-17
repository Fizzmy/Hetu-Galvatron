"""Ray Cluster event handlers"""

import gradio as gr
from ..state import state
from profiling import get_available_node_configs


def initialize_ray_cluster(address: str):
    """Initialize Ray cluster and return cluster message"""
    try:
        msg = state.ray_cluster.initialize(address if address else None)
        
        if "successfully" in msg or "already" in msg:
            state.cluster_info = state.ray_cluster.get_cluster_info()
            # Initialize task queue with cluster manager
            from core import RayTaskQueue
            state.ray_task_manager = RayTaskQueue(
                ray_cluster_manager=state.ray_cluster
            )
            return state.ray_cluster.format_cluster_info(state.cluster_info)
        else:
            return msg
    except Exception as e:
        import traceback
        return f"Failed to initialize Ray: {e}\n\n{traceback.format_exc()}"


def generate_config_choices():
    """Generate config choices from cluster_info and auto-select the largest"""
    config_choices = []
    selected_config = None
    if state.cluster_info:
        gpus_per_node = state.cluster_info.get('gpus_per_node', 0)
        max_nodes = state.cluster_info.get('num_nodes', 1)
        if gpus_per_node > 0:
            configs = get_available_node_configs(gpus_per_node, max_nodes)
            config_choices = [
                (f"{nodes} Nodes x {gpus} GPUs = {nodes * gpus} Total GPUs", f"{nodes}_{gpus}")
                for nodes, gpus in configs
            ]
            
            # Auto-select the largest configuration (max nodes * gpus)
            if configs:
                max_config = max(configs, key=lambda x: x[0] * x[1])
                selected_config = f"{max_config[0]}_{max_config[1]}"
    return gr.update(choices=config_choices, value=selected_config)


def init_ray_cluster_with_config(address: str):
    """Initialize Ray cluster and update config dropdown (for both hardware and search tabs)"""
    cluster_msg = initialize_ray_cluster(address)
    if "Ray Cluster Initialized" in cluster_msg:
        config_update = generate_config_choices()
        search_config_update = generate_config_choices()  # Same update for search tab
        state.init_hardware_profile_managers()
    else:
        config_update = gr.update(choices=[], value=None)
        search_config_update = gr.update(choices=[], value=None)
    return cluster_msg, config_update, search_config_update


def init_ray_cluster_with_config_all(address: str):
    """Initialize Ray cluster and update config dropdown (for hardware, search, and training tabs)"""
    cluster_msg = initialize_ray_cluster(address)
    if "Ray Cluster Initialized" in cluster_msg:
        config_update = generate_config_choices()
        search_config_update = generate_config_choices()  # Same update for search tab
        training_config_update = generate_config_choices()  # Same update for training tab
        state.init_hardware_profile_managers()
    else:
        config_update = gr.update(choices=[], value=None)
        search_config_update = gr.update(choices=[], value=None)
        training_config_update = gr.update(choices=[], value=None)
    return cluster_msg, config_update, search_config_update, training_config_update

