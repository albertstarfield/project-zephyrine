#!/bin/bash

# --- Configuration ---
VENV_DIR="venv"
REQUIREMENTS_FILE="backend/requirements.txt"
START_SCRIPT="backend/start.sh"
RUN_SCRIPT="run.sh"

# --- Functions ---

# Check if a suitable Python 3 version (3.11 or 3.12) is available
check_python3() {
  # Check if the specific version is available
  if ! command -v python3.11 &> /dev/null && ! command -v python3.12 &> /dev/null; then
    echo "Error: Neither Python 3.11 nor Python 3.12 could be found."
    echo "Please install Python 3.11 or 3.12 and ensure they are in your PATH."
    exit 1
  fi

  # Set PYTHON to the appropriate command
  if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
  elif command -v python3.12 &> /dev/null; then
    PYTHON=python3.12
  fi

  # No need for further version checks here as we've already confirmed availability
}

# Check if virtual environment exists
# Use the $PYTHON variable for subsequent commands
check_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating it now..."
    "$PYTHON" -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to create virtual environment."
      exit 1
    fi
  fi
}

# Activate the virtual environment
activate_venv() {
  source "$VENV_DIR/bin/activate"
}


install_requirements() {
  echo "Installing requirements..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements."
    exit 1
  fi
}

# --- Main Script ---

# Check for Python 3.11 or 3.12
check_python3

# Check and create virtual environment if needed
check_venv

# Activate the virtual environment
activate_venv

# Install requirements
install_requirements

# Start the backend
echo "Starting backend..."
bash "$START_SCRIPT"

# Run the main script
echo "Running application..."
bash "$RUN_SCRIPT"

# Deactivate the virtual environment (optional)
# deactivate

echo "Application finished."