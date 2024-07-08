#!/bin/bash

# Define the required directories
required_dirs=(systemCore launchcontrol launcher-builder-control documentation)

# Check if all required directories exist
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: $dir directory is missing"
        exit 1
    fi
done

# Record the current working directory
cwd=$(pwd)
echo "Current working directory: $cwd"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# Create a virtual environment (venv) in the current working directory
python3 -m venv _venv

# Activate the virtual environment
source _venv/bin/activate

# Run the launch control script using the virtual environment
python3 ./launchcontrol/coreRuntimeManagement.py
