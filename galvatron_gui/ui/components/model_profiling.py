"""Model Profiling tab UI components"""

import gradio as gr
from ..handlers import model_profiling as handlers


def create_model_profiling_tab():
    """Create the Model Profiling tab"""
    with gr.Tab("Model Profiling"):
        gr.Markdown("""
### Model Profiling
Profile model computation and memory characteristics for different model configurations
""")
        
        # Step 1: Select Model Configuration
        with gr.Accordion("1️⃣ Select Model Configuration", open=True):
            gr.Markdown("### Step 1: Select Model Type and Size")
            
            with gr.Row():
                model_type_dropdown = gr.Dropdown(
                    label="Select Model Type",
                    choices=[],
                    interactive=True,
                    info="Select model type (e.g., llama, qwen, gpt)"
                )
                
                model_size_dropdown = gr.Dropdown(
                    label="Select Model Size",
                    choices=[],
                    interactive=True,
                    info="Select model size (e.g., 7b, 13b, 30b)"
                )
                
                refresh_model_list_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm", scale=0)
            
            load_model_btn = gr.Button("Load Model Configuration", variant="secondary")
            
            # Model structure display (auto-loaded after model selection)
            model_structure_display = gr.Markdown("")
            
            # Editable model parameters
            with gr.Accordion("📝 Model Parameters (Editable)", open=True):
                gr.Markdown("### Edit Model Configuration Parameters")
                
                # Core parameters (for LLaMA, GPT, BERT, etc.)
                with gr.Row():
                    hidden_size_input = gr.Number(label="Hidden Size", value=None, interactive=True, precision=0)
                    num_hidden_layers_input = gr.Number(label="Num Hidden Layers", value=None, interactive=True, precision=0)
                    num_attention_heads_input = gr.Number(label="Num Attention Heads", value=None, interactive=True, precision=0)
                    num_key_value_heads_input = gr.Number(label="Num Key Value Heads", value=None, interactive=True, precision=0)
                
                with gr.Row():
                    intermediate_size_input = gr.Number(label="Intermediate Size", value=None, interactive=True, precision=0)
                    max_position_embeddings_input = gr.Number(label="Max Position Embeddings", value=None, interactive=True, precision=0)
                    vocab_size_input = gr.Number(label="Vocab Size", value=None, interactive=True, precision=0)
        
        # Step 2: Computation Profile
        with gr.Accordion("2️⃣ Computation Profile", open=True):
            gr.Markdown("### Computation Profile Configuration")
            
            with gr.Row():
                comp_profile_mode_dropdown = gr.Dropdown(
                    label="Profile Mode",
                    choices=[
                        ("Static", "static"),
                        ("Batch", "batch"),
                        ("Sequence", "sequence")
                    ],
                    value="static",
                    interactive=True,
                    visible=False,
                    info="Profile mode: static (fixed), batch (vary batch size), or sequence (vary sequence length)"
                )
            
            # Profile parameters (dynamic based on mode)
            with gr.Accordion("📝 Profile Parameters", open=True, visible=False) as comp_params_accordion:
                # Static mode parameters
                with gr.Row(visible=True) as comp_static_params:
                    comp_static_batch_size = gr.Number(label="Batch Size", value=1, interactive=True, precision=0, minimum=1)
                    comp_static_seq_length = gr.Number(label="Sequence Length", value=2048, interactive=True, precision=0, minimum=1)
                
                # Batch mode parameters
                with gr.Row(visible=False) as comp_batch_params:
                    comp_batch_min_batch_size = gr.Number(label="Min Batch Size", value=1, interactive=True, precision=0, minimum=1)
                    comp_batch_max_batch_size = gr.Number(label="Max Batch Size", value=8, interactive=True, precision=0, minimum=1)
                    comp_batch_batch_size_step = gr.Number(label="Batch Size Step", value=1, interactive=True, precision=0, minimum=1)
                    comp_batch_seq_length = gr.Number(label="Sequence Length", value=2048, interactive=True, precision=0, minimum=1)
                
                # Sequence mode parameters
                with gr.Row(visible=False) as comp_sequence_params:
                    comp_sequence_min_seq_length = gr.Number(label="Min Sequence Length", value=512, interactive=True, precision=0, minimum=1)
                    comp_sequence_max_seq_length = gr.Number(label="Max Sequence Length", value=2048, interactive=True, precision=0, minimum=1)
                    comp_sequence_seq_length_step = gr.Number(label="Sequence Length Step", value=128, interactive=True, precision=0, minimum=1)
                    comp_sequence_batch_size = gr.Number(label="Batch Size", value=1, interactive=True, precision=0, minimum=1)
            
            comp_load_config_btn = gr.Button("Load Config", variant="primary", visible=False)
            
            comp_profile_missing_only_checkbox = gr.Checkbox(
                label="Profile Missing Data Only",
                value=False,
                visible=False,
                info="Only profile missing computation data keys"
            )
            
            with gr.Row():
                comp_profile_status_display = gr.Markdown("")
                comp_refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm", scale=0, visible=False)
            
            comp_submit_profile_btn = gr.Button("Start Computation Profile", variant="primary", visible=False)
            
            comp_profile_output = gr.Markdown("")
        
        # Step 3: Memory Profile
        with gr.Accordion("3️⃣ Memory Profile", open=True):
            gr.Markdown("### Memory Profile Configuration")
            
            with gr.Row():
                mem_profile_mode_dropdown = gr.Dropdown(
                    label="Profile Mode",
                    choices=[
                        ("Static", "static")
                    ],
                    value="static",
                    interactive=True,
                    visible=False,
                    info="Memory profiling only supports static mode"
                )
            
            # Profile parameters (static mode only for memory)
            with gr.Accordion("📝 Profile Parameters", open=True, visible=False) as mem_params_accordion:
                with gr.Row(visible=True) as mem_static_params:
                    mem_static_batch_size = gr.Number(label="Batch Size", value=1, interactive=True, precision=0, minimum=1)
                    mem_static_seq_length = gr.Number(label="Sequence Length", value=2048, interactive=True, precision=0, minimum=1)
            
            # Layer number parameters
            with gr.Row(visible=False) as mem_layernum_row:
                mem_layernum_min = gr.Number(label="Min Layer Number", value=1, interactive=True, precision=0, minimum=1)
                mem_layernum_max = gr.Number(label="Max Layer Number", value=2, interactive=True, precision=0, minimum=1)
            
            mem_load_config_btn = gr.Button("Load Config", variant="primary", visible=False)
            
            mem_profile_missing_only_checkbox = gr.Checkbox(
                label="Profile Missing Data Only",
                value=False,
                visible=False,
                info="Only profile missing memory data keys"
            )
            
            with gr.Row():
                mem_profile_status_display = gr.Markdown("")
                mem_refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm", scale=0, visible=False)
            
            mem_submit_profile_btn = gr.Button("Start Memory Profile", variant="primary", visible=False)
            
            mem_profile_output = gr.Markdown("")
        
        # Event handlers
        # Refresh model list button
        refresh_model_list_btn.click(
            handlers.refresh_model_list,
            inputs=[],
            outputs=[model_type_dropdown]
        )
        
        # When model type changes, update model size dropdown
        model_type_dropdown.change(
            handlers.update_model_sizes,
            inputs=[model_type_dropdown],
            outputs=[model_size_dropdown]
        )
        
        load_model_btn.click(
            handlers.load_model_config,
            inputs=[model_type_dropdown, model_size_dropdown],
            outputs=[
                model_structure_display,
                # Computation profile UI
                comp_profile_mode_dropdown,
                comp_params_accordion,
                comp_static_params,
                comp_batch_params,
                comp_sequence_params,
                comp_load_config_btn,
                # Memory profile UI
                mem_profile_mode_dropdown,
                mem_params_accordion,
                mem_layernum_row,
                mem_load_config_btn,
                # Parameter inputs
                hidden_size_input,
                num_hidden_layers_input,
                num_attention_heads_input,
                num_key_value_heads_input,
                intermediate_size_input,
                max_position_embeddings_input,
                vocab_size_input,
            ]
        )
        
        # Update parameter visibility based on profile mode
        def update_comp_params_visibility(mode):
            """Update computation profile parameter visibility based on mode"""
            if mode == "static":
                return (
                    gr.update(visible=True),  # comp_params_accordion
                    gr.update(visible=True),  # comp_static_params
                    gr.update(visible=False),  # comp_batch_params
                    gr.update(visible=False),  # comp_sequence_params
                    gr.update(visible=True),  # comp_load_config_btn
                )
            elif mode == "batch":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            elif mode == "sequence":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
        
        comp_profile_mode_dropdown.change(
            update_comp_params_visibility,
            inputs=[comp_profile_mode_dropdown],
            outputs=[
                comp_params_accordion,
                comp_static_params,
                comp_batch_params,
                comp_sequence_params,
                comp_load_config_btn,
            ]
        )
        
        # Create State components for constant values
        comp_profile_type_state = gr.State("computation")
        mem_profile_type_state = gr.State("memory")
        mem_profile_mode_state = gr.State("static")
        comp_layernum_min_state = gr.State(None)
        comp_layernum_max_state = gr.State(None)
        
        # Computation profile handlers
        comp_load_config_btn.click(
            handlers.load_profile_config,
            inputs=[
                comp_profile_type_state,
                model_type_dropdown,
                model_size_dropdown,
                comp_profile_mode_dropdown,
                comp_static_batch_size,
                comp_static_seq_length,
                comp_batch_min_batch_size,
                comp_batch_max_batch_size,
                comp_batch_batch_size_step,
                comp_batch_seq_length,
                comp_sequence_min_seq_length,
                comp_sequence_max_seq_length,
                comp_sequence_seq_length_step,
                comp_sequence_batch_size,
                comp_layernum_min_state,
                comp_layernum_max_state,
                hidden_size_input,
                num_hidden_layers_input,
                num_attention_heads_input,
                num_key_value_heads_input,
                intermediate_size_input,
                max_position_embeddings_input,
                vocab_size_input,
            ],
            outputs=[
                comp_profile_status_display,
                comp_profile_missing_only_checkbox,
                comp_refresh_status_btn,
                comp_submit_profile_btn,
                comp_profile_output,
            ]
        )
        
        comp_refresh_status_btn.click(
            handlers.refresh_profile_status,
            inputs=[
                model_type_dropdown,
                model_size_dropdown,
                comp_profile_type_state,
                comp_profile_mode_dropdown,
            ],
            outputs=[comp_profile_status_display]
        )
        
        comp_submit_profile_btn.click(
            handlers.submit_model_profile,
            inputs=[
                comp_profile_type_state,
                model_type_dropdown,
                model_size_dropdown,
                comp_profile_mode_dropdown,
                comp_static_batch_size,
                comp_static_seq_length,
                comp_batch_min_batch_size,
                comp_batch_max_batch_size,
                comp_batch_batch_size_step,
                comp_batch_seq_length,
                comp_sequence_min_seq_length,
                comp_sequence_max_seq_length,
                comp_sequence_seq_length_step,
                comp_sequence_batch_size,
                comp_layernum_min_state,
                comp_layernum_max_state,
                hidden_size_input,
                num_hidden_layers_input,
                num_attention_heads_input,
                intermediate_size_input,
                max_position_embeddings_input,
                vocab_size_input,
                comp_profile_missing_only_checkbox,
            ],
            outputs=[comp_profile_output]
        )
        
        # Memory profile handlers
        # Create State components for None values (unused parameters for memory)
        mem_batch_min_state = gr.State(None)
        mem_batch_max_state = gr.State(None)
        mem_batch_step_state = gr.State(None)
        mem_batch_seq_state = gr.State(None)
        mem_seq_min_state = gr.State(None)
        mem_seq_max_state = gr.State(None)
        mem_seq_step_state = gr.State(None)
        mem_seq_batch_state = gr.State(None)
        
        mem_load_config_btn.click(
            handlers.load_profile_config,
            inputs=[
                mem_profile_type_state,
                model_type_dropdown,
                model_size_dropdown,
                mem_profile_mode_state,
                mem_static_batch_size,
                mem_static_seq_length,
                mem_batch_min_state,
                mem_batch_max_state,
                mem_batch_step_state,
                mem_batch_seq_state,
                mem_seq_min_state,
                mem_seq_max_state,
                mem_seq_step_state,
                mem_seq_batch_state,
                mem_layernum_min,
                mem_layernum_max,
                hidden_size_input,
                num_hidden_layers_input,
                num_attention_heads_input,
                num_key_value_heads_input,
                intermediate_size_input,
                max_position_embeddings_input,
                vocab_size_input,
            ],
            outputs=[
                mem_profile_status_display,
                mem_profile_missing_only_checkbox,
                mem_refresh_status_btn,
                mem_submit_profile_btn,
                mem_profile_output,
            ]
        )
        
        mem_refresh_status_btn.click(
            handlers.refresh_profile_status,
            inputs=[
                model_type_dropdown,
                model_size_dropdown,
                mem_profile_type_state,
                mem_profile_mode_dropdown,
            ],
            outputs=[mem_profile_status_display]
        )
        
        mem_submit_profile_btn.click(
            handlers.submit_model_profile,
            inputs=[
                mem_profile_type_state,
                model_type_dropdown,
                model_size_dropdown,
                mem_profile_mode_state,
                mem_static_batch_size,
                mem_static_seq_length,
                mem_batch_min_state,
                mem_batch_max_state,
                mem_batch_step_state,
                mem_batch_seq_state,
                mem_seq_min_state,
                mem_seq_max_state,
                mem_seq_step_state,
                mem_seq_batch_state,
                mem_layernum_min,
                mem_layernum_max,
                hidden_size_input,
                num_hidden_layers_input,
                num_attention_heads_input,
                intermediate_size_input,
                max_position_embeddings_input,
                vocab_size_input,
                mem_profile_missing_only_checkbox,
            ],
            outputs=[mem_profile_output]
        )
        
        
        # Note: Model list will be initialized when the tab is first accessed
        # We'll handle this in the app.load event
    
    return model_type_dropdown

