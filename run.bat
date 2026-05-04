@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set "VENV_DIR=.venv"
set "REQUIREMENTS=requirements.txt"
set "NOTEBOOK=compare.ipynb"
set "APP=app.py"

:: 1. Check and create the virtual environment
if not exist "%VENV_DIR%\" (
    echo Creating virtual environment in '%VENV_DIR%'...
    python -m venv "%VENV_DIR%"
) else (
    echo Virtual environment '%VENV_DIR%' already exists.
)

:: 2. Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: 3. Install dependencies
if exist "%REQUIREMENTS%" (
    echo Installing/updating dependencies...
    python -m pip install --upgrade pip -q
    pip install -r "%REQUIREMENTS%" -q
) else (
    echo Warning: %REQUIREMENTS% not found. Skipping pip install.
)

:: 4. Run the notebook 3 times
cd notebooks
if exist "%NOTEBOOK%" (
    echo Running notebook '%NOTEBOOK%' for each dataset
    set i=1
    for %%f in (3 18 44) do (
        echo  -^> Execution run !i! of 3...
        papermill "%NOTEBOOK%" "%NOTEBOOK%" -p d %%f
        set /a i+=1
    )
    echo Notebook executions complete.
) else (
    echo Error: Notebook '%NOTEBOOK%' not found! Skipping...
)
cd ..

:: 5. Launch the Streamlit app
if exist "%APP%" (
    echo Launching %APP%...
    streamlit run "%APP%"
) else (
    echo Error: %APP% not found in the current directory.
)