#!/bin/bash

echo "==================================================="
echo "     Fintech Credit Scoring - Pipeline Manager"
echo "==================================================="
echo ""

# ---------------------------------------------------------
# 1. Virtual Environment & Dependencies
# ---------------------------------------------------------
read -p "1. Do you want to use a virtual environment (.venv)? (Y/N): " USE_VENV

if [[ "${USE_VENV,,}" == "y" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo " -> Creating virtual environment in '.venv'..."
        python3 -m venv .venv
    else
        echo " -> Virtual environment '.venv' already exists."
    fi
    
    echo " -> Activating virtual environment..."
    source .venv/bin/activate

    if [[ -f "requirements.txt" ]]; then
        echo " -> Installing dependencies in .venv..."
        python3 -m pip install --upgrade pip -q
        python3 -m pip install -r requirements.txt -q
    else
        echo " -> Warning: requirements.txt not found. Skipping pip install."
    fi
else
    if [[ -f "requirements.txt" ]]; then
        echo " -> Installing dependencies in current global environment..."
        python3 -m pip install --upgrade pip -q
        python3 -m pip install -r requirements.txt -q
    else
        echo " -> Warning: requirements.txt not found. Skipping pip install."
    fi
fi
echo ""

# ---------------------------------------------------------
# 2. Train Models
# ---------------------------------------------------------
read -p "2. Do you want to train the models? (WARNING: it is very slow)(Y/N): " TRAIN_MODELS

if [[ "${TRAIN_MODELS,,}" == "y" ]]; then
    echo " -> Scanning 'src' directory for model scripts..."
    if ls src/*model.py 1> /dev/null 2>&1; then
        for f in src/*model.py; do
            echo "   -> Running $f..."
            python3 "$f"
        done
        echo " -> Model training complete."
    else
        echo " -> Warning: No files ending with 'model.py' were found in the 'src' folder."
    fi
fi
echo ""

# ---------------------------------------------------------
# 3. Evaluation Results
# ---------------------------------------------------------
read -p "3. Do you want to create evaluation results? (Y/N): " RUN_EVAL

if [[ "${RUN_EVAL,,}" == "y" ]]; then
    echo " -> Generating evaluation results..."
    if [[ -f "notebooks/compare.ipynb" ]]; then
        cd notebooks || exit
        
        # Use dynamic output names to prevent overwriting
        for f in 3 18 44; do
            echo "   -> Executing compare.ipynb for $f features..."
            papermill "compare.ipynb" "compare_${f}_features.ipynb" -p d "$f"
        done
        cd ..
        echo " -> Evaluation complete."
    else
        echo " -> Warning: notebooks/compare.ipynb not found."
    fi
fi
echo ""

# ---------------------------------------------------------
# 4. Launch Demo App
# ---------------------------------------------------------
read -p "4. Do you want to launch the demo app? (Y/N): " LAUNCH_APP

if [[ "${LAUNCH_APP,,}" == "y" ]]; then
    echo " -> Launching Streamlit app..."
    if [[ -f "app.py" ]]; then
        streamlit run app.py
    else
        echo " -> Warning: app.py not found in the root directory."
    fi
fi

echo ""
echo "Pipeline execution finished."