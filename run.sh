#!/bin/bash

echo "==================================================="
echo "     Fintech Credit Scoring - Pipeline Manager"
echo "==================================================="
echo ""

# ---------------------------------------------------------
# 1. Virtual Environment & Dependencies
# ---------------------------------------------------------
read -p "1. Do you want to use a virtual environment (.venv)? (NOTE: This will create a new virtual environment, SKIP if you are already using one) (Y/N by default): " USE_VENV

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
        echo " -> Installing dependencies in current environment..."
        python3 -m pip install --upgrade pip -q
        python3 -m pip install -r requirements.txt -q
    else
        echo " -> Warning: requirements.txt not found. Skipping pip install."
    fi
fi
echo ""

# ---------------------------------------------------------
# 2. Download Raw Dataset
# ---------------------------------------------------------
read -p "2. Do you want to download the raw dataset? (NOTE: It is not needed if you don't want to preprocess and create data splits) (Y/N by default): " DOWNLOAD_DATA

if [[ "${DOWNLOAD_DATA,,}" == "y" ]]; then
    echo " -> Running data download script..."
    python3 src/download_data.py
fi
echo ""

# ---------------------------------------------------------
# 3. Create Data Split
# ---------------------------------------------------------
read -p "3. Do you want to preprocess raw dataset and create data splits? (NOTE: The repository already contains preprocessed data) (Y/N by default): " CREATE_SPLIT

if [[ "${CREATE_SPLIT,,}" == "y" ]]; then
    if [[ -f "notebooks/preprocess.ipynb" ]]; then
        cd notebooks || exit

        echo "   -> Executing preprocess.ipynb"
        papermill "preprocess.ipynb" "preprocess_executed.ipynb"
        
        echo "   -> Cleaning up executed notebook..."
        rm "preprocess_executed.ipynb"

        cd ..
        echo " -> Preprocessing and data split creation complete."
    else
        echo " -> Warning: notebooks/preprocess.ipynb not found."
    fi
fi
echo ""

# ---------------------------------------------------------
# 4. Delete Raw Dataset
# ---------------------------------------------------------
if [[ -f "data/raw/data.xls" ]]; then
    read -p "4. Do you want to delete the raw dataset? (y/N): " DELETE_DATA

    if [[ "${DELETE_DATA,,}" == "y" ]]; then
        echo " -> Deleting the raw dataset file..."
        rm "data/raw/data.xls"
        echo " -> Successfully deleted data/raw/data.xls"
    fi
else
    echo "4. Delete Raw Dataset: Skipped (data/raw/data.xls not found)"
fi
echo ""

# ---------------------------------------------------------
# 5. Train Models
# ---------------------------------------------------------
read -p "5. Do you want to train the models? (WARNING: it is very slow)(Y/N by default): " TRAIN_MODELS

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
# 6. Evaluation Results
# ---------------------------------------------------------
read -p "6. Do you want to create evaluation results? (Y/N by default): " RUN_EVAL

if [[ "${RUN_EVAL,,}" == "y" ]]; then
    echo " -> Generating evaluation results..."
    if [[ -f "notebooks/compare.ipynb" ]]; then
        cd notebooks || exit
        
        for f in 3 18 44; do
            echo "   -> Executing compare.ipynb for $f features..."
            papermill "compare.ipynb" "compare_${f}_features.ipynb" -p d "$f"
            
            echo "   -> Cleaning up executed evaluation notebook..."
            rm "compare_${f}_features.ipynb"
        done

        cd ..
        echo " -> Evaluation complete."
    else
        echo " -> Warning: notebooks/compare.ipynb not found."
    fi
fi
echo ""

# ---------------------------------------------------------
# 7. Fairness Analysis
# ---------------------------------------------------------
read -p "7. Do you want to run fairness analysis? (Y/N by default): " RUN_FAIRNESS

if [[ "${RUN_FAIRNESS,,}" == "y" ]]; then
    echo " -> Running fairness analysis..."

    if [[ -f "src/fairness_analysis.py" ]]; then
        python3 src/fairness_analysis.py
    else
        echo " -> Warning: src/fairness_analysis.py not found."
    fi

    if [[ -f "notebooks/fairness.ipynb" ]]; then
        cd notebooks || exit

        echo "   -> Executing fairness.ipynb"
        papermill "fairness.ipynb" "fairness_executed.ipynb"

        echo "   -> Cleaning up executed notebook..."
        rm "fairness_executed.ipynb"

        cd ..
        echo " -> Fairness analysis complete."
    else
        echo " -> Warning: notebooks/fairness.ipynb not found."
    fi
fi
echo ""

# ---------------------------------------------------------
# 8. Launch Demo App
# ---------------------------------------------------------
read -p "8. Do you want to launch the demo app? (Y/N by default): " LAUNCH_APP

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
