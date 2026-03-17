"""Task Monitor tab UI components"""

import gradio as gr
from ..handlers import task_monitor as handlers


def create_task_monitor_tab():
    """Create the Task Monitor tab"""
    with gr.Tab("Task Monitor") as task_monitor_tab:
        gr.Markdown("### Monitor All Tasks")
        gr.Markdown("💡 **Tip:** Click '🔄 Refresh' button to update status and trigger pending tasks")
        
        with gr.Row():
            refresh_tasks_button = gr.Button("🔄 Refresh", variant="primary", scale=1)
            cancel_task_id_input = gr.Textbox(label="Task ID to Cancel", placeholder="Enter Task ID", scale=2)
            cancel_task_button = gr.Button("❌ Cancel", variant="stop", scale=1)
            retry_task_id_input = gr.Textbox(label="Task ID to Retry", placeholder="Enter Task ID", scale=2)
            retry_task_button = gr.Button("🔄 Retry", variant="secondary", scale=1)
        
        task_list_output = gr.Markdown("No tasks running.")
        
        with gr.Row():
            cancel_output = gr.Textbox(label="Cancel Status", lines=1)
            retry_output = gr.Textbox(label="Retry Status", lines=1)
        
        # Manual refresh (triggers scheduling)
        refresh_tasks_button.click(
            handlers.refresh_task_list,
            inputs=[],
            outputs=[task_list_output]
        )
        
        cancel_task_button.click(
            handlers.cancel_task_by_id,
            inputs=[cancel_task_id_input],
            outputs=[cancel_output]
        )
        
        retry_task_button.click(
            handlers.retry_task_by_id,
            inputs=[retry_task_id_input],
            outputs=[retry_output]
        )
    
    # Return task_list_output for auto-refresh setup in app.py
    return task_list_output

