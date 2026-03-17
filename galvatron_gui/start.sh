#!/bin/bash

echo "========================================="
echo "Galvatron Profiling GUI"
echo "========================================="
echo ""

# Check if Ray is running
if ! ray status > /dev/null 2>&1; then
    echo "Starting Ray head node..."
    ray start --head --dashboard-host=0.0.0.0 --port=6379
    sleep 3
else
    echo "Ray is already running"
fi

echo ""
echo "Launching Galvatron Profiling GUI..."
echo "GUI:       http://localhost:7860"
echo "Dashboard: http://localhost:8265"
echo ""

python main.py

