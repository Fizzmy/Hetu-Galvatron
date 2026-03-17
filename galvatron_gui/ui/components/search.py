"""Search tab UI components"""

import gradio as gr
from ..handlers import search as handlers


def create_search_tab():
    """Create the Parallelism Strategy Search tab"""
    with gr.Tab("Strategy Search"):
        gr.Markdown("""
### Parallelism Strategy Search
Search for optimal parallelism strategies based on profiled data
""")
        
        # Step 1: Select Model and Load Profiling Results
        with gr.Accordion("1️⃣ Select Model & Check Profiling Results", open=True):
            gr.Markdown("### Step 1: Select Model and Load Profiling Results")
            
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
                info="Select cluster configuration for search (nodes x GPUs/node)"
            )
            
            load_profiling_btn = gr.Button("Load Profiling Results", variant="primary")
            
            profiling_status_display = gr.Markdown("")
        
        # Profiling config selection (visible after loading, outside accordion)
        with gr.Accordion("📋 Select Profiling Configurations", open=True, visible=False) as profiling_config_accordion:
            gr.Markdown("Select which profiling configurations to use for the search:")
            with gr.Row():
                comp_profile_dropdown = gr.Dropdown(
                    label="⏱️ Computation Profiling Mode",
                    choices=[],
                    interactive=True,
                    info="Select computation profiling mode"
                )
                
                mem_profile_dropdown = gr.Dropdown(
                    label="💾 Memory Profiling Mode",
                    choices=[],
                    interactive=True,
                    info="Select memory profiling mode"
                )
        
        # Step 2: Configure Search Space
        with gr.Accordion("2️⃣ Configure Search Parameters", open=True, visible=False) as search_params_accordion:
            gr.Markdown("### Search Parameters")
            
            # Batch Size Parameter (fixed batch size)
            with gr.Row():
                batch_size = gr.Number(label="Batch Size", value=16, interactive=True, precision=0, minimum=1)
                settle_chunk = gr.Number(label="Settle Chunk", value=-1, interactive=True, precision=0, minimum=-1)
            
            # Memory Constraint
            with gr.Row():
                memory_constraint = gr.Number(label="Memory Constraint (GB)", value=36, interactive=True, precision=0, minimum=1)
                seq_length = gr.Number(label="Sequence Length", value=2048, interactive=True, precision=0, minimum=1)
            
            # Search Space Configuration
            with gr.Row():
                search_space = gr.Dropdown(
                    label="Search Space",
                    choices=["full", "dp+tp", "dp+pp", "3d", "dp", "sdp", "tp", "pp"],
                    value="full",
                    interactive=True,
                    info="Parallelism optimization type"
                )
                
                sp_space = gr.Dropdown(
                    label="Sequence Parallel Space",
                    choices=["tp+sp", "tp", "sp"],
                    value="tp+sp",
                    interactive=True,
                    info="Sequence parallelism type"
                )
            
            # Parallelism Limits
            with gr.Row():
                max_tp_deg = gr.Number(label="Max TP Degree", value=8, interactive=True, precision=0, minimum=1)
                max_pp_deg = gr.Number(label="Max PP Degree", value=16, interactive=True, precision=0, minimum=1)
            
            # Disable Options
            with gr.Accordion("Advanced: Disable Parallelism Options", open=False):
                with gr.Row():
                    disable_dp = gr.Checkbox(label="Disable DP", value=False)
                    disable_tp = gr.Checkbox(label="Disable TP", value=False)
                    disable_pp = gr.Checkbox(label="Disable PP", value=False)
                    disable_sdp = gr.Checkbox(label="Disable SDP", value=False)
                
                with gr.Row():
                    disable_ckpt = gr.Checkbox(label="Disable Checkpoint", value=False)
                    disable_vtp = gr.Checkbox(label="Disable VocabTP", value=False)
                    disable_tp_consec = gr.Checkbox(label="Disable TP Consecutive", value=True)
                
                with gr.Row():
                    no_global_memory_buffer = gr.Checkbox(label="No Global Memory Buffer", value=False, info="Disable global memory buffer for Megatron-SP")
                    no_async_grad_reduce = gr.Checkbox(label="No Async Grad Reduce", value=False, info="Disable async grad reduce (Zero3 memory cost when chunk > 1)")
            
            # Other Options
            with gr.Row():
                pipeline_type = gr.Dropdown(
                    label="Pipeline Type",
                    choices=["gpipe", "pipedream_flush"],
                    value="pipedream_flush",
                    interactive=True
                )
                
                default_dp_type = gr.Dropdown(
                    label="Default DP Type",
                    choices=["ddp", "zero2"],
                    value="zero2",
                    interactive=True
                )
                
                mixed_precision = gr.Dropdown(
                    label="Mixed Precision",
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    interactive=True
                )
            
            with gr.Row():
                fine_grained_mode = gr.Checkbox(label="Fine-Grained Mode", value=True)
                sequence_parallel = gr.Checkbox(label="Sequence Parallel", value=True)
        
        # Step 3: Start Search
        with gr.Accordion("3️⃣ Start Strategy Search", open=True, visible=False) as search_action_accordion:
            search_status_display = gr.Markdown("")
            
            with gr.Row():
                start_search_btn = gr.Button("🔍 Start Search", variant="primary", size="lg")
                refresh_search_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
            
            search_output = gr.Markdown("")
        
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
        
        load_profiling_btn.click(
            handlers.load_profiling_results,
            inputs=[model_type_dropdown, model_size_dropdown, cluster_config_dropdown],
            outputs=[
                profiling_status_display,
                profiling_config_accordion,
                comp_profile_dropdown,
                mem_profile_dropdown,
                search_params_accordion,
                search_action_accordion,
            ]
        )
        
        start_search_btn.click(
            handlers.submit_search,
            inputs=[
                model_type_dropdown,
                model_size_dropdown,
                cluster_config_dropdown,
                comp_profile_dropdown,
                mem_profile_dropdown,
                batch_size,
                settle_chunk,
                memory_constraint,
                seq_length,
                search_space,
                sp_space,
                max_tp_deg,
                max_pp_deg,
                disable_dp,
                disable_tp,
                disable_pp,
                disable_sdp,
                disable_ckpt,
                disable_vtp,
                disable_tp_consec,
                no_global_memory_buffer,
                no_async_grad_reduce,
                pipeline_type,
                default_dp_type,
                mixed_precision,
                fine_grained_mode,
                sequence_parallel,
            ],
            outputs=[search_output, search_status_display]
        )
        
        refresh_search_status_btn.click(
            handlers.refresh_search_status,
            inputs=[model_type_dropdown, model_size_dropdown],
            outputs=[search_status_display]
        )
    
    return model_type_dropdown, cluster_config_dropdown
