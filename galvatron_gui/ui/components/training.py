"""Training tab UI components"""

import gradio as gr
from ..handlers import training as handlers


def create_training_tab():
    """Create the Model Training tab"""
    with gr.Tab("Model Training"):
        gr.Markdown("""
### Distributed Model Training
Train models using optimized parallelism strategies
""")
        
        # Step 1: Select Model and Cluster
        with gr.Accordion("1️⃣ Select Model & Cluster Configuration", open=True):
            gr.Markdown("### Step 1: Select Model and Cluster")
            
            with gr.Row():
                model_type_dropdown = gr.Dropdown(
                    label="Select Model Type",
                    choices=[],
                    interactive=True,
                    info="Select model type (e.g., llama, qwen)"
                )
                
                model_size_dropdown = gr.Dropdown(
                    label="Select Model Size",
                    choices=[],
                    interactive=True,
                    info="Select model size (e.g., 7b, 13b)"
                )
                
                refresh_model_list_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm", scale=0)
            
            # Cluster configuration selection
            cluster_config_dropdown = gr.Dropdown(
                label="Select Cluster Configuration",
                choices=[],
                interactive=True,
                info="Select cluster configuration for training (nodes x GPUs/node)"
            )
            
            load_strategies_btn = gr.Button("Load Available Strategies", variant="primary")
            
            strategy_status_display = gr.Markdown("")
        
        # Step 2: Select Parallelism Strategy
        with gr.Accordion("2️⃣ Select Parallelism Strategy", open=True, visible=False) as strategy_accordion:
            gr.Markdown("### Select Strategy from Search Results")
            
            strategy_dropdown = gr.Dropdown(
                label="Available Parallelism Strategies",
                choices=[],
                interactive=True,
                info="Select a strategy configuration"
            )
            
            strategy_details_display = gr.Markdown("")
        
        # Step 3: Configure Training Hyperparameters
        with gr.Accordion("3️⃣ Configure Training Hyperparameters", open=True, visible=False) as training_params_accordion:
            gr.Markdown("### Training Parameters")
            
            # Basic Training Parameters
            with gr.Row():
                train_iters = gr.Number(label="Train Iterations", value=100, interactive=True, precision=0, minimum=1)
            
            # Learning Rate Parameters
            gr.Markdown("#### Learning Rate Configuration")
            with gr.Row():
                lr = gr.Number(label="Learning Rate", value=1.25e-6, interactive=True)
                min_lr = gr.Number(label="Min Learning Rate", value=1.25e-7, interactive=True)
                lr_warmup_fraction = gr.Number(label="LR Warmup Fraction", value=0.1, interactive=True, minimum=0, maximum=1)
            
            with gr.Row():
                lr_decay_style = gr.Dropdown(
                    label="LR Decay Style",
                    choices=["cosine", "linear", "constant"],
                    value="cosine",
                    interactive=True
                )
            
            # Optimizer Parameters
            gr.Markdown("#### Optimizer Configuration")
            with gr.Row():
                adam_weight_decay = gr.Number(label="Adam Weight Decay", value=0.01, interactive=True, minimum=0, maximum=1)
                adam_beta1 = gr.Number(label="Adam Beta1", value=0.9, interactive=True, minimum=0, maximum=1)
                adam_beta2 = gr.Number(label="Adam Beta2", value=0.95, interactive=True, minimum=0, maximum=1)
            
            with gr.Row():
                adam_eps = gr.Number(label="Adam Epsilon", value=1.0e-5, interactive=True)
                async_grad_reduce = gr.Checkbox(
                    label="Async Gradient Reduce",
                    value=True,
                    interactive=True,
                    info="Enable asynchronous gradient reduction"
                )
            
            # Data Parameters
            gr.Markdown("#### Data Configuration")
            with gr.Row():
                data_path = gr.Dropdown(
                    label="Data Path",
                    choices=[],
                    interactive=True,
                    info="Select training dataset directory (from configured paths in gui_config.json)"
                )
                split = gr.Textbox(
                    label="Data Split",
                    value="949,50,1",
                    interactive=True,
                    info="Train/valid/test split ratios (comma-separated)"
                )
            
            with gr.Row():
                seq_length = gr.Number(
                    label="Sequence Length",
                    value=4096,
                    interactive=True,
                    precision=0,
                    minimum=1,
                    info="Maximum sequence length for training"
                )
            
            with gr.Row():
                tokenizer_model = gr.Dropdown(
                    label="Tokenizer Model",
                    choices=[],
                    interactive=True,
                    info="Select tokenizer model directory (from configured paths in gui_config.json)"
                )
                refresh_paths_btn = gr.Button("🔄 Refresh Paths", variant="secondary", size="sm")
            
            # Checkpoint Parameters
            gr.Markdown("#### Checkpoint Configuration (Optional)")
            with gr.Row():
                enable_load_checkpoint = gr.Checkbox(
                    label="Enable Load Checkpoint",
                    value=False,
                    interactive=True,
                    info="Load checkpoint from a previous training run"
                )
                enable_save_checkpoint = gr.Checkbox(
                    label="Enable Save Checkpoint",
                    value=False,
                    interactive=True,
                    info="Save checkpoints during training"
                )
            
            with gr.Row():
                load = gr.Textbox(
                    label="Load Checkpoint Path",
                    placeholder="/path/to/checkpoint",
                    visible=False,
                    interactive=True,
                    info="Enter checkpoint directory path to load from"
                )
                
                save = gr.Textbox(
                    label="Save Checkpoint Path",
                    placeholder="/path/to/save/checkpoint",
                    visible=False,
                    interactive=True,
                    info="Enter directory path to save checkpoints"
                )
            
            with gr.Row():
                save_interval = gr.Number(
                    label="Save Interval",
                    value=100,
                    visible=False,
                    interactive=True,
                    precision=0,
                    minimum=1,
                    info="Save checkpoint every N iterations"
                )
            
            # Update visibility based on checkboxes
            def update_load_checkpoint_visibility(enable):
                return gr.update(visible=enable)
            
            def update_save_checkpoint_visibility(enable):
                return gr.update(visible=enable), gr.update(visible=enable)
            
            enable_load_checkpoint.change(
                update_load_checkpoint_visibility,
                inputs=[enable_load_checkpoint],
                outputs=[load]
            )
            
            enable_save_checkpoint.change(
                update_save_checkpoint_visibility,
                inputs=[enable_save_checkpoint],
                outputs=[save, save_interval]
            )
        
        # Step 4: Start Training
        with gr.Accordion("4️⃣ Start Training", open=True, visible=False) as training_action_accordion:
            training_status_display = gr.Markdown("")
            
            with gr.Row():
                start_training_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                refresh_training_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                auto_refresh_checkbox = gr.Checkbox(
                    label="⏱️ Auto Refresh (5s)",
                    value=False,
                    interactive=True,
                    info="Automatically refresh training status and plots every 5 seconds"
                )
            
            training_output = gr.Markdown("")
            with gr.Row():
                loss_plot = gr.Image(label=None, visible=True)
                time_plot = gr.Image(label=None, visible=True)
            
            # Timer for auto-refresh (always running, but only refreshes when checkbox is enabled)
            training_refresh_timer = gr.Timer(value=5)
        
        # Event handlers
        refresh_model_list_btn.click(
            handlers.refresh_model_list,
            inputs=[],
            outputs=[model_type_dropdown]
        )
        
        model_type_dropdown.change(
            handlers.update_model_sizes,
            inputs=[model_type_dropdown],
            outputs=[model_size_dropdown]
        )
        
        # Load strategies and paths together
        def load_strategies_and_paths(model_type, model_size, cluster_config):
            """Load strategies and paths"""
            strategy_results = handlers.load_available_strategies(model_type, model_size, cluster_config)
            data_update = handlers.scan_directory_for_paths("", path_type="dataset")
            tokenizer_update = handlers.scan_directory_for_paths("", path_type="tokenizer")
            return (*strategy_results, data_update, tokenizer_update)
        
        load_strategies_btn.click(
            load_strategies_and_paths,
            inputs=[model_type_dropdown, model_size_dropdown, cluster_config_dropdown],
            outputs=[
                strategy_status_display,
                strategy_accordion,
                strategy_dropdown,
                training_params_accordion,
                training_action_accordion,
                data_path,
                tokenizer_model,
            ]
        )
        
        # Refresh paths independently
        refresh_paths_btn.click(
            lambda: (
                handlers.scan_directory_for_paths("", path_type="dataset"),
                handlers.scan_directory_for_paths("", path_type="tokenizer")
            ),
            inputs=[],
            outputs=[data_path, tokenizer_model]
        )
        
        strategy_dropdown.change(
            handlers.display_strategy_details,
            inputs=[strategy_dropdown],
            outputs=[strategy_details_display]
        )
        
        start_training_btn.click(
            handlers.submit_training,
            inputs=[
                model_type_dropdown,
                model_size_dropdown,
                cluster_config_dropdown,
                strategy_dropdown,
                train_iters,
                lr,
                min_lr,
                lr_warmup_fraction,
                lr_decay_style,
                adam_weight_decay,
                adam_beta1,
                adam_beta2,
                adam_eps,
                async_grad_reduce,
                seq_length,
                data_path,
                split,
                tokenizer_model,
                enable_load_checkpoint,
                load,
                enable_save_checkpoint,
                save,
                save_interval,
            ],
            outputs=[training_output, training_status_display]
        )
        
        refresh_training_status_btn.click(
            handlers.refresh_training_status,
            inputs=[model_type_dropdown, model_size_dropdown],
            outputs=[training_status_display, loss_plot, time_plot]
        )
        
        # Auto-refresh timer: only refresh when checkbox is enabled
        def auto_refresh_training_status(enable_auto_refresh, model_type, model_size):
            """Auto-refresh training status if enabled"""
            if enable_auto_refresh and model_type and model_size:
                return handlers.refresh_training_status(model_type, model_size)
            else:
                # Return current values (no update) when auto-refresh is disabled
                return gr.update(), gr.update(), gr.update()
        
        # Timer tick event: refresh training status and plots every 5 seconds
        # Timer always runs, but only refreshes when checkbox is enabled
        training_refresh_timer.tick(
            auto_refresh_training_status,
            inputs=[auto_refresh_checkbox, model_type_dropdown, model_size_dropdown],
            outputs=[training_status_display, loss_plot, time_plot]
        )
    
    return model_type_dropdown, cluster_config_dropdown


