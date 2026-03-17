"""Hardware Profiling event handlers"""

import time
import json
from typing import Dict

from ..state import state
from profiling import BandwidthProfileManager, OverlapCoefficientManager


def refresh_overlap_status():
    """Refresh overlap coefficient status"""
    state.overlap_manager._load_data()
    return state.overlap_manager.get_status()


def submit_overlap_profile():
    """Submit overlap coefficient profiling task to pending queue"""
    return state.overlap_manager.submit_task()


def check_bandwidth_profiles_status(num_nodes: int, num_gpus_per_node: int) -> Dict[str, bool]:
    """Check status of 4 bandwidth profile types"""
    status = {
        "allreduce": False,  # dp/sdp (allreduce_consec0)
        "p2p": False,        # pp
        "tp_allreduce": False,  # tp (allreduce_consec1)
        "sp_all2all": False   # sp (all2all)
    }
    
    try:
        managers = state.get_or_create_bandwidth_managers(num_nodes, num_gpus_per_node)
        
        # Check allreduce (dp/sdp)
        managers['dp']._load_data()
        status["allreduce"] = managers['dp'].is_completed()
        
        # Check p2p (pp)
        managers['pp']._load_data()
        status["p2p"] = managers['pp'].is_completed()
        
        # Check allreduce (tp)
        managers['tp']._load_data()
        status["tp_allreduce"] = managers['tp'].is_completed()
        
        # Check all2all (sp)
        managers['sp']._load_data()
        status["sp_all2all"] = managers['sp'].is_completed()
    except Exception as e:
        import traceback
        print(f"Error checking bandwidth profiles status: {e}\n{traceback.format_exc()}")
    
    return status


def get_bandwidth_profiles_display(num_nodes: int, num_gpus_per_node: int) -> str:
    """Get formatted display for bandwidth profiles status"""
    status = check_bandwidth_profiles_status(num_nodes, num_gpus_per_node)
    
    profiles = [
        ("Data Parallel / Sequence Data Parallel (DP/SDP)", "allreduce", "Unconsecutive AllReduce Bandwidth"),
        ("Pipeline Parallel (PP)", "p2p", "P2P Bandwidth"),
        ("Tensor Parallel (TP)", "tp_allreduce", "Consecutive AllReduce Bandwidth"),
        ("Sequence Parallel (SP)", "sp_all2all", "Consecutive All2All Bandwidth")
    ]
    
    display = "### 并行策略带宽 Profile 状态\n\n"
    display += f"**集群配置:** {num_nodes} Nodes × {num_gpus_per_node} GPUs/Node\n\n"
    display += "| 并行策略 | 带宽类型 | 状态 |\n"
    display += "|---------|---------|------|\n"
    
    for strategy, key, bandwidth_type in profiles:
        checkmark = "✅" if status[key] else "❌"
        display += f"| {strategy} | {bandwidth_type} | {checkmark} |\n"
    
    return display


def get_bandwidth_profile_details(num_nodes: int, num_gpus_per_node: int) -> str:
    """Get detailed profile results for all bandwidth profiles"""
    details = []
    
    # Helper function to format profile details
    def format_profile_details(manager: BandwidthProfileManager, title: str):
        """Format profile details from manager"""
        try:
            manager._load_data()
            is_completed = manager.is_completed()
            
            if is_completed:
                details.append(f"## ✅ {title}\n\n")
            else:
                details.append(f"## ❌ {title}\n\n")
            
            details.append(f"**File:** `{manager.config_file.name}`\n\n")
            
            # Show existing data
            if manager.data:
                details.append("**Existing Results:**\n\n")
                # Show first 10 items
                items = list(manager.data.items())[:10]
                for key, value in items:
                    if isinstance(value, (int, float)):
                        details.append(f"- `{key}`: {value:.4f}\n")
                    else:
                        details.append(f"- `{key}`: {value}\n")
                if len(manager.data) > 10:
                    details.append(f"\n... and {len(manager.data) - 10} more items\n")
                details.append("\n")
            
            # Show missing data
            if not is_completed and hasattr(manager, 'need_to_profile'):
                missing_keys = [key for key in manager.need_to_profile if key not in manager.data]
                if missing_keys:
                    details.append("**Missing Data:**\n\n")
                    for key in missing_keys:
                        details.append(f"- `{key}`\n")
                    details.append(f"\n**Total:** {len(missing_keys)} missing items out of {len(manager.need_to_profile)} required\n\n")
            
        except Exception as e:
            details.append(f"## ❌ {title}\n\n**Error reading file:** {str(e)}\n\n")
    
    try:
        managers = state.get_or_create_bandwidth_managers(num_nodes, num_gpus_per_node)
        
        # AllReduce Bandwidth (DP/SDP) - unconsecutive
        format_profile_details(managers['dp'], "Unconsecutive AllReduce Bandwidth (DP/SDP)")
        
        # P2P Bandwidth (PP)
        format_profile_details(managers['pp'], "P2P Bandwidth (PP)")
        
        # AllReduce (TP) - consecutive
        format_profile_details(managers['tp'], "Consecutive AllReduce (TP)")
        
        # All2All (SP) - consecutive
        format_profile_details(managers['sp'], "Consecutive All2All (SP)")
    except Exception as e:
        import traceback
        details.append(f"❌ Error loading profile details: {str(e)}\n\n{traceback.format_exc()}\n\n")
    
    if not details:
        return "❌ No profile results found. Please run profiling tasks first."
    
    return "".join(details)


def load_config_and_show_strategies(config_str: str):
    """Load config and show parallel strategies status"""
    import gradio as gr
    
    if not config_str:
        return "Please select a configuration", gr.update(visible=False, choices=[]), gr.update(visible=False), gr.update(visible=False), ""
    
    try:
        nodes, gpus = map(int, config_str.split('_'))
        # Store config info in state for later use
        state.current_hw_config_nodes = nodes
        state.current_hw_config_gpus = gpus
        
        # Get bandwidth profiles status display
        display = get_bandwidth_profiles_display(nodes, gpus)
        
        # Get detailed results
        details = get_bandwidth_profile_details(nodes, gpus)
        
        # Create dropdown choices for profile types
        profile_choices = [
            ("All", "all"),
            ("Unconsecutive AllReduce (DP/SDP)", "allreduce_bandwidth"),
            ("P2P (PP)", "p2p_bandwidth"),
            ("Consecutive AllReduce (TP)", "tp_allreduce"),
            ("Consecutive All2All (SP)", "sp_all2all"),
        ]
        
        return (
            display,
            gr.update(visible=True, choices=profile_choices, value=None),  # profile_type_dropdown
            gr.update(visible=True),  # profile_missing_only_checkbox
            gr.update(visible=True),  # submit_profile_btn
            details
        )
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", gr.update(visible=False, choices=[]), gr.update(visible=False), gr.update(visible=False), ""


def refresh_profiles_status():
    """Refresh profile status and details"""
    if not hasattr(state, 'current_hw_config_nodes') or not hasattr(state, 'current_hw_config_gpus'):
        return "Please select a configuration first", ""
    
    try:
        nodes = state.current_hw_config_nodes
        gpus = state.current_hw_config_gpus
        
        # Refresh status display
        display = get_bandwidth_profiles_display(nodes, gpus)
        
        # Get detailed results
        details = get_bandwidth_profile_details(nodes, gpus)
        
        return display, details
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", ""


def submit_bandwidth_profile_from_ui(profile_type_value: str, profile_missing_only: bool):
    """Submit a bandwidth profile task from UI
    
    Args:
        profile_type_value: Selected profile type value from dropdown (all, allreduce_bandwidth, p2p_bandwidth, tp_allreduce, sp_all2all)
        profile_missing_only: Whether to only profile missing data
    """
    if not profile_type_value:
        return "Please select a profile type first."
    
    if not hasattr(state, 'current_hw_config_nodes') or not hasattr(state, 'current_hw_config_gpus'):
        return "Please select a cluster configuration first."
    
    try:
        num_nodes = state.current_hw_config_nodes
        num_gpus_per_node = state.current_hw_config_gpus
        
        # Get or create BandwidthProfileManager instances from state
        managers = state.get_or_create_bandwidth_managers(num_nodes, num_gpus_per_node)
        
        # If "all" is selected, submit tasks for all profile types
        if profile_type_value == "all":
            results = []
            profile_configs = [
                ("dp", "Unconsecutive AllReduce (DP/SDP)"),
                ("pp", "P2P (PP)"),
                ("tp", "Consecutive AllReduce (TP)"),
                ("sp", "Consecutive All2All (SP)"),
            ]
            
            for parallel_strategy, description in profile_configs:
                manager = managers[parallel_strategy]
                result = manager.submit_task(profile_missing_only=profile_missing_only)
                results.append(f"**{description}:**\n{result}\n")
            
            return "\n".join(results)
        
        # Map dropdown value to profile_type and parallel_strategy
        if profile_type_value == "allreduce_bandwidth":
            parallel_strategy = "dp"
        elif profile_type_value == "p2p_bandwidth":
            parallel_strategy = "pp"
        elif profile_type_value == "tp_allreduce":
            parallel_strategy = "tp"
        elif profile_type_value == "sp_all2all":
            parallel_strategy = "sp"
        else:
            return f"❌ Error: Unknown profile type: {profile_type_value}"
        
        # Submit task for selected profile type
        manager = managers[parallel_strategy]
        return manager.submit_task(profile_missing_only=profile_missing_only)
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


def submit_bandwidth_profile(profile_type: str, comm_type: str = None, profile_missing_only: bool = False):
    """Submit a bandwidth profile task (legacy function for backward compatibility)
    
    Args:
        profile_type: Type of profile (allreduce_bandwidth, p2p_bandwidth, sp_time)
        comm_type: Communication type for sp_time (allreduce or all2all)
        profile_missing_only: If True, only profile missing data keys
    """
    if not hasattr(state, 'current_hw_config_nodes') or not hasattr(state, 'current_hw_config_gpus'):
        return "Please select a cluster configuration first."
    
    try:
        num_nodes = state.current_hw_config_nodes
        num_gpus_per_node = state.current_hw_config_gpus
        
        # Determine profile_type and parallel_strategy based on input
        if profile_type == "allreduce_bandwidth":
            profile_type_arg = "allreduce"
            parallel_strategy = "dp"
        elif profile_type == "p2p_bandwidth":
            profile_type_arg = "p2p"
            parallel_strategy = "pp"
        elif profile_type == "sp_time":
            if comm_type == "allreduce":
                profile_type_arg = "allreduce"
                parallel_strategy = "tp"
            elif comm_type == "all2all":
                profile_type_arg = "all2all"
                parallel_strategy = "sp"
            else:
                return "❌ Error: comm_type must be 'allreduce' or 'all2all' for sp_time"
        else:
            return f"❌ Error: Unknown profile type: {profile_type}"
        
        # Get or create BandwidthProfileManager instance from state and submit task
        managers = state.get_or_create_bandwidth_managers(num_nodes, num_gpus_per_node)
        manager = managers[parallel_strategy]
        return manager.submit_task(profile_missing_only=profile_missing_only)
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

