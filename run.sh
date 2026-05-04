#!/bin/bash

# --- Configuration ---
VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
NOTEBOOK="compare.ipynb" 
APP="app.py"

# 1. Check and create the virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
if [ -f "$REQUIREMENTS" ]; then
    echo "Installing/updating dependencies..."
    pip install --upgrade pip -q
    pip install -r "$REQUIREMENTS" -q
else
    echo "Warning: $REQUIREMENTS not found. Skipping pip install."
fi

# 4. Run the notebook 3 times
declare -a n_features
n_features=(3 18 44)
cd notebooks
if [ -f "$NOTEBOOK" ]; then
    echo "Running notebook '$NOTEBOOK' for each dataset"
    for i in {1..3}; do
        echo " -> Execution run $i of 3..."
        papermill "$NOTEBOOK" "$NOTEBOOK" -p d ${n_features[$((i-1))]}
    done
    echo "Notebook executions complete."
else
    echo "Error: Notebook '$NOTEBOOK' not found! Skipping..."
fi
cd ..

# 5. Launch the Streamlit app
if [ -f "$APP" ]; then
    echo "Launching $APP..."
    streamlit run "$APP"
else
    echo "Error: $APP not found in the current directory."
fi