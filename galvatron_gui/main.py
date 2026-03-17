#!/usr/bin/env python3
"""
Galvatron Profiling GUI
Main entry point for the application
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui import create_app

demo = create_app()

def main():
    """Launch the Gradio application"""
    print("=" * 60)
    print("Galvatron Profiling GUI")
    print("=" * 60)
    print("\nStarting application...")
    print("Make sure Ray cluster is running:")
    print("  $ ray start --head --dashboard-host=0.0.0.0")
    print("\nAccess the GUI at: http://localhost:7860")
    print("=" * 60)
    print()
    

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

