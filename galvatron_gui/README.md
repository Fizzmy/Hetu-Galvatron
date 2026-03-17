# Galvatron Profiling GUI

A Ray-based distributed profiling system for Galvatron with a Gradio web interface.

## Project Structure

```
galvatron_gui/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
│
├── core/                # Core functionality
│   ├── __init__.py
│   ├── config.py        # Task configurations
│   ├── ray_manager.py   # Ray cluster management
│   └── task_queue.py    # Task queue and execution
│
├── profiling/           # Profiling modules
│   ├── __init__.py
│   ├── hardware.py      # Hardware profiling
│   └── analyzer.py      # Profile analysis
│
├── ui/                  # User interface
│   ├── __init__.py
│   ├── app.py           # Gradio application
│   └── handlers.py      # Event handlers
│
└── data/                # Data storage
    └── task_configs/    # Task configuration files
```

## Quick Start

### 1. Install Dependencies

```bash
cd galvatron_gui
pip install -r requirements.txt
```

### 2. Start Ray Cluster

```bash
# Single node
ray start --head --dashboard-host=0.0.0.0

# Multi-node (on head node)
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Multi-node (on worker nodes)
ray start --address=<head-ip>:6379
```

### 3. Launch GUI

```bash
python main.py
```

Access the GUI at: http://localhost:7860

## Features

### Hardware Profiling
- **Overlap Coefficient**: Test compute-communication overlap
- **Cluster Bandwidth**: Profile AllReduce, P2P, and Sequence Parallel communication

### Workflow
1. Initialize Ray cluster
2. Select cluster configuration (automatically detected from Ray)
3. Choose specific test items to profile
4. Monitor task progress in real-time

### Key Features
- **Automatic Configuration**: Detects cluster setup from Ray
- **Granular Control**: Select specific test keys to profile
- **Real-time Monitoring**: Track task status and node assignment
- **Result Display**: View profiling results directly in the GUI

## Ray Cluster Management

### Check Ray Status
```bash
ray status
```

### Stop Ray
```bash
ray stop
```

### Ray Dashboard
Access at: http://localhost:8265

## Development

### Module Overview

- **core**: Ray integration, task management, configuration
- **profiling**: Hardware/model profiling logic and analysis
- **ui**: Gradio interface and event handling

### Adding New Features

1. Add business logic to `core/` or `profiling/`
2. Create handler functions in `ui/handlers.py`
3. Add UI components in `ui/app.py`

## Troubleshooting

### Ray Connection Issues
- Ensure Ray is started: `ray start --head --dashboard-host=0.0.0.0`
- Check Ray status: `ray status`
- Restart Ray: `ray stop && ray start --head --dashboard-host=0.0.0.0`

### Port Conflicts
- GUI port: Change `server_port` in `main.py`
- Ray port: Use `--port` flag with `ray start`

## License

See parent project LICENSE file.
