"""Hardware Profiling tab UI components"""

import gradio as gr
from ..handlers import hardware_profiling as handlers


def create_hardware_profiling_tab(cluster_config_dropdown=None):
    """Create the Hardware Profiling tab"""
    with gr.Tab("Hardware Profiling"):
        gr.Markdown("""
### Hardware Profiling
Two-step hardware performance testing: Overlap Coefficient + Cluster Bandwidth
""")
        
        # Part 1: Overlap Coefficient
        with gr.Accordion("1️⃣ Overlap Coefficient Profile", open=True):
            gr.Markdown("Test compute-communication overlap coefficient (cluster-independent)")
            
            overlap_status_display = gr.Markdown("")
            
            with gr.Row():
                check_overlap_btn = gr.Button("Check Status", size="sm")
                submit_overlap_btn = gr.Button("Start Profile", variant="primary")
            
            overlap_output = gr.Markdown("")
        
        # Part 2: Cluster Bandwidth
        with gr.Accordion("2️⃣ Cluster Bandwidth Profile", open=True):
            gr.Markdown("### Step 1: Select Cluster Configuration")
            
            if cluster_config_dropdown is None:
                cluster_config_dropdown = gr.Dropdown(
                    label="Select Configuration",
                    choices=[],
                    interactive=True,
                    info="Select configuration to profile (nodes x GPUs/node)"
                )
            
            load_config_btn = gr.Button("Load Configuration and Show Parallel Strategies", variant="secondary")
            
            gr.Markdown("---")
            gr.Markdown("### Step 2: Parallel Strategy Bandwidth Profiles")
            
            with gr.Row():
                bandwidth_profiles_display = gr.Markdown("")
                refresh_profiles_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm", scale=0)
            
            with gr.Row():
                profile_type_dropdown = gr.Dropdown(
                    label="Select Profile Type",
                    choices=[],
                    interactive=True,
                    visible=False,
                    info="Select bandwidth profile type to run"
                )
                profile_missing_only_checkbox = gr.Checkbox(
                    label="Profile Missing Data Only",
                    value=False,
                    visible=False,
                    info="Only profile missing data keys"
                )
                submit_profile_btn = gr.Button("Start Profile", variant="primary", visible=False)
            
            hw_profile_output = gr.Markdown("")
            
            # Detailed results accordion
            with gr.Accordion("📊 View Detailed Profile Results", open=False):
                profile_details_display = gr.Markdown("")
        
        # Event handlers
        check_overlap_btn.click(
            handlers.refresh_overlap_status,
            inputs=[],
            outputs=[overlap_status_display]
        )
        
        submit_overlap_btn.click(
            handlers.submit_overlap_profile,
            inputs=[],
            outputs=[overlap_output]
        )
        
        load_config_btn.click(
            handlers.load_config_and_show_strategies,
            inputs=[cluster_config_dropdown],
            outputs=[bandwidth_profiles_display, profile_type_dropdown, profile_missing_only_checkbox, submit_profile_btn, profile_details_display]
        )
        
        refresh_profiles_btn.click(
            handlers.refresh_profiles_status,
            inputs=[],
            outputs=[bandwidth_profiles_display, profile_details_display]
        )
        
        submit_profile_btn.click(
            handlers.submit_bandwidth_profile_from_ui,
            inputs=[profile_type_dropdown, profile_missing_only_checkbox],
            outputs=[hw_profile_output]
        )
    
    return (
        cluster_config_dropdown, 
        handlers.load_config_and_show_strategies, 
        overlap_status_display,
        (
            bandwidth_profiles_display, 
            profile_type_dropdown,
            profile_missing_only_checkbox,
            submit_profile_btn,
            profile_details_display
        )
    )

