@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo      Fintech Credit Scoring - Pipeline Manager
echo ===================================================
echo.

:: ---------------------------------------------------------
:: 1. Virtual Environment & Dependencies
:: ---------------------------------------------------------
set /p USE_VENV="1. Do you want to use a virtual environment (.venv)? (Y/N): "

if /I "!USE_VENV!"=="Y" (
    if not exist ".venv\" (
        echo  -^> Creating virtual environment in '.venv'...
        python -m venv .venv
    ) else (
        echo  -^> Virtual environment '.venv' already exists.
    )
    
    echo  -^> Activating virtual environment...
    call .venv\Scripts\activate.bat

    if exist "requirements.txt" (
        echo  -^> Installing dependencies in .venv...
        ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    ) else (
        echo  -^> Warning: requirements.txt not found. Skipping pip install.
    )
) else (
    if exist "requirements.txt" (
        echo  -^> Installing dependencies in current environment...
        python -m pip install --upgrade pip -q
        pip install -r requirements.txt -q
    ) else (
        echo  -^> Warning: requirements.txt not found. Skipping pip install.
    )
)
echo.

:: ---------------------------------------------------------
:: 2. Train Models
:: ---------------------------------------------------------
set /p TRAIN_MODELS="2. Do you want to train the models? (WARNING: it is very slow)(Y/N): "

if /I "!TRAIN_MODELS!"=="Y" (
    echo  -^> Scanning 'src' directory for model scripts...
    if exist "src\*model.py" (
        for %%f in (src\*model.py) do (
            echo    -^> Running %%f...
            python "%%f"
        )
        echo  -^> Model training complete.
    ) else (
        echo  -^> Warning: No files ending with 'model.py' were found in the 'src' folder.
    )
)
echo.

:: ---------------------------------------------------------
:: 3. Evaluation Results
:: ---------------------------------------------------------
set /p RUN_EVAL="3. Do you want to create evaluation results? (Y/N): "

if /I "!RUN_EVAL!"=="Y" (
    echo  -^> Generating evaluation results...
    if exist "notebooks\compare.ipynb" (
        cd notebooks
        
        :: Using your original papermill loop to evaluate all 3 dataset sizes
        for %%f in (3 18 44) do (
            echo    -^> Executing compare.ipynb for %%f features...
            papermill "compare.ipynb" "compare.ipynb" -p d %%f
        )
        cd ..
        echo  -^> Evaluation complete.
    ) else (
        echo  -^> Warning: notebooks\compare.ipynb not found.
    )
)
echo.

:: ---------------------------------------------------------
:: 4. Launch Demo App
:: ---------------------------------------------------------
set /p LAUNCH_APP="4. Do you want to launch the demo app? (Y/N): "

if /I "!LAUNCH_APP!"=="Y" (
    echo  -^> Launching Streamlit app...
    if exist "app.py" (
        streamlit run app.py
    ) else (
        echo  -^> Warning: app.py not found in the root directory.
    )
)

echo.
echo Pipeline execution finished.
pause