"""Main Gradio application"""

import gradio as gr
from . import styles
from .components import create_ray_cluster_tab, create_hardware_profiling_tab, create_model_profiling_tab, create_task_monitor_tab, create_search_tab, create_training_tab
from .handlers import ray_cluster as ray_handlers, hardware_profiling as hw_handlers, model_profiling as model_handlers, task_monitor as task_handlers, search as search_handlers, training as training_handlers


def create_app():
    """Create and configure the Gradio application"""
    
    with gr.Blocks(title="Galvatron Profiling GUI", theme=gr.themes.Soft(), css=styles.CUSTOM_CSS) as app:
        
        gr.Markdown("# Galvatron Cluster Profiling")
        gr.Markdown("Ray-based distributed profiling system")
        
        with gr.Tabs():
            # Tab 1: Ray Cluster
            ray_address, cluster_output, init_button = create_ray_cluster_tab()
            
            # Tab 2: Hardware Profiling
            cluster_config_dropdown, load_config_fn, overlap_status_display, hw_outputs = create_hardware_profiling_tab()
            
            # Tab 3: Model Profiling
            model_dropdown = create_model_profiling_tab()
            
            # Tab 4: Strategy Search (creates its own cluster config dropdown)
            search_model_dropdown, search_cluster_config_dropdown = create_search_tab()
            
            # Tab 5: Model Training (creates its own cluster config dropdown)
            training_model_dropdown, training_cluster_config_dropdown = create_training_tab()
            
            # Tab 6: Task Monitor
            task_list_output = create_task_monitor_tab()
        
        # Set up init_button.click to also update cluster_config_dropdown (hw, search, and training)
        init_button.click(
            ray_handlers.init_ray_cluster_with_config_all,
            inputs=[ray_address],
            outputs=[cluster_output, cluster_config_dropdown, search_cluster_config_dropdown, training_cluster_config_dropdown]
        )
        
        # Auto-initialize Ray cluster and UI components when app loads
        def init_app(address: str):
            """Initialize app: connect to Ray, update config dropdown, and refresh overlap status"""
            cluster_msg, config_update, search_config_update, training_config_update = ray_handlers.init_ray_cluster_with_config_all(address)
            # Refresh model config cache and initialize model type list
            model_handlers.refresh_model_config_cache()
            model_types = model_handlers.get_available_model_types()
            model_type_update = gr.update(choices=model_types) if model_types else gr.update(choices=[])
            return cluster_msg, config_update, search_config_update, training_config_update, model_type_update, model_type_update, model_type_update
        
        app.load(
            init_app,
            inputs=[ray_address],
            outputs=[cluster_output, cluster_config_dropdown, search_cluster_config_dropdown, training_cluster_config_dropdown, model_dropdown, search_model_dropdown, training_model_dropdown]
        )
        
        # Auto-load config when dropdown value changes (including initial selection)
        cluster_config_dropdown.change(
            load_config_fn,
            inputs=[cluster_config_dropdown],
            outputs=hw_outputs
        )
        
        # Auto-refresh task list every 0.5 seconds using Timer
        task_refresh_timer = gr.Timer(value=5)
        task_refresh_timer.tick(
            task_handlers.refresh_task_list,
            inputs=[],
            outputs=[task_list_output]
        )
    
    return app


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
