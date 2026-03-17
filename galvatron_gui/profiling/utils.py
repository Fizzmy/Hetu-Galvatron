from typing import List, Tuple
from pathlib import Path
from .hardware import HARDWARE_CONFIGS_DIR


def get_available_node_configs(gpus_per_node: int, max_nodes: int = 16) -> List[Tuple[int, int]]:
    """Generate available node configurations based on cluster capacity"""
    configs = []
    
    # Single node configurations
    for gpus in [1, 2, 4, 8]:
        if gpus <= gpus_per_node:
            configs.append((1, gpus))
    
    # Multi-node configurations
    for nodes in [2, 4, 8, 16]:
        if nodes <= max_nodes:
            configs.append((nodes, gpus_per_node))
    
    return configs


def format_node_configs_display(gpus_per_node: int, max_nodes: int, configs: List[Tuple[int, int]]) -> str:
    """Format node configurations for display"""
    display = f"### Ray Cluster Information\n\n"
    display += f"- **GPUs per Node:** {gpus_per_node}\n"
    display += f"- **Max Nodes:** {max_nodes}\n\n"
    display += "### Available Configurations\n\n"
    display += "| Config | Nodes | GPUs/Node | Total GPUs | Note |\n"
    display += "|--------|-------|-----------|------------|------|\n"
    
    for i, (nodes, gpus) in enumerate(configs, 1):
        total = nodes * gpus
        note = "Partial node usage" if nodes == 1 and gpus < gpus_per_node else "Full usage"
        
        # Check if profile exists
        has_profile = any([
            (HARDWARE_CONFIGS_DIR / f"allreduce_consec0_bandwidth_{nodes}nodes_{gpus}gpus_per_node.json").exists(),
            (HARDWARE_CONFIGS_DIR / f"allreduce_consec1_bandwidth_{nodes}nodes_{gpus}gpus_per_node.json").exists(),
            (HARDWARE_CONFIGS_DIR / f"p2p_bandwidth_{nodes}nodes_{gpus}gpus_per_node.json").exists(),
            (HARDWARE_CONFIGS_DIR / f"all2all_bandwidth_{nodes}nodes_{gpus}gpus_per_node.json").exists()
        ])
        
        status = "✅" if has_profile else "❌"
        display += f"| Config {i} {status} | {nodes} | {gpus} | {total} | {note} |\n"
    
    display += "\n💡 ✅ = Has profile data, ❌ = Not yet profiled\n"
    
    return display
