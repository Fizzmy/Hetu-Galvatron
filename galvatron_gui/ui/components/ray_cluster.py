"""Ray Cluster tab UI components"""

import gradio as gr


def create_ray_cluster_tab():
    """Create the Ray Cluster tab"""
    with gr.Tab("Ray Cluster"):
        gr.Markdown("### Initialize Ray Cluster")
        
        with gr.Row():
            ray_address = gr.Textbox(
                label="Ray Address (optional)",
                placeholder="ray://head-node-ip:10001 or leave empty for auto",
                value=""
            )
        
        init_button = gr.Button("Initialize Ray Cluster", variant="primary")
        cluster_output = gr.Markdown("")
        
        # Note: init_button.click will be set up in app.py to also update cluster_config_dropdown
    
    return ray_address, cluster_output, init_button

