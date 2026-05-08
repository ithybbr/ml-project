@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo      Fintech Credit Scoring - Pipeline Manager
echo ===================================================
echo.

:: ---------------------------------------------------------
:: 1. Virtual Environment & Dependencies
:: ---------------------------------------------------------
set /p USE_VENV="1. Do you want to use a virtual environment (.venv)? (NOTE: This will create a new virtual environment, SKIP if you are already using one) (Y/N by default): "

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
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
    ) else (
        echo  -^> Warning: requirements.txt not found. Skipping pip install.
    )
) else (
    if exist "requirements.txt" (
        echo  -^> Installing dependencies in current environment...
        python -m pip install --upgrade pip -q
        python -m pip install -r requirements.txt -q
    ) else (
        echo  -^> Warning: requirements.txt not found. Skipping pip install.
    )
)
echo.

:: ---------------------------------------------------------
:: 2. Download Raw Dataset
:: ---------------------------------------------------------
if not exist "data\raw\data.xls" (
    set /p DOWNLOAD_DATA="2. Do you want to download the raw dataset? (NOTE: It is not needed if you don't want to preprocess and create data splits) (Y/N by default): "

if /I "!DOWNLOAD_DATA!"=="Y" (
    echo  -^> Running data download script...
    python src/download_data.py
)
echo.
) else (
    echo  -^> Raw dataset already exists at data\raw\data.xls. Skipping download.
    echo.
)
:: ---------------------------------------------------------
:: 3. Create Data Split
:: ---------------------------------------------------------
if exist "data\raw\data.xls" (
    set /p CREATE_SPLIT="3. Do you want to preprocess raw dataset and create data splits? (Y/N by default): "

    if /I "!CREATE_SPLIT!"=="Y" (
            if exist "notebooks\preprocess.ipynb" (
                cd notebooks

                echo    -^> Executing preprocess.ipynb
                papermill "preprocess.ipynb" "preprocess_executed.ipynb"
                
                echo    -^> Cleaning up executed notebook...
                del "preprocess_executed.ipynb"

                cd ..
                echo  -^> Preprocessing and data split creation complete.
        ) else (
            echo  -^> Warning: preprocess.ipynb is not found.
        )
    )
    echo.
) else (
    echo 3. Create Data Split: Skipped ^(data\raw\data.xls not found^)
    echo.
)

:: ---------------------------------------------------------
:: 4. Delete Raw Dataset
:: ---------------------------------------------------------
if exist "data\raw\data.xls" (
    set /p DELETE_DATA="4. Do you want to delete the raw dataset? (y/N): "

    if /I "!DELETE_DATA!"=="Y" (
        echo  -^> Deleting the raw dataset file...
        del "data\raw\data.xls"
        echo  -^> Successfully deleted data\raw\data.xls
    )
) else (
    echo 4. Delete Raw Dataset: Skipped ^(data\raw\data.xls not found^)
)
echo.

:: ---------------------------------------------------------
:: 5. Train Models
:: ---------------------------------------------------------
set /p TRAIN_MODELS="5. Do you want to train the models? (WARNING: it is very slow) (Y/N by default): "

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
:: 6. Evaluation Results
:: ---------------------------------------------------------
set /p RUN_EVAL="6. Do you want to create evaluation results? (Y/N by default): "

if /I "!RUN_EVAL!"=="Y" (
    echo  -^> Generating evaluation results...
    if exist "notebooks\compare.ipynb" (
        cd notebooks
        
        for %%f in (3 18 44) do (
            echo    -^> Executing compare.ipynb for %%f features...
            papermill "compare.ipynb" "compare_%%f_features.ipynb" -p d %%f
            
            echo    -^> Cleaning up executed evaluation notebook...
            del "compare_%%f_features.ipynb"
        )

        cd ..
        echo  -^> Evaluation complete.
    ) else (
        echo  -^> Warning: notebooks\compare.ipynb not found.
    )
    if exist "src\fairness_analysis.py" (
        echo  -^> Running fairness analysis...
        python src/fairness_analysis.py
    )
    if exist "notebooks\fairness.ipynb" (
        cd notebooks

        echo    -^> Executing fairness.ipynb
        papermill "fairness.ipynb" "fairness_executed.ipynb"

        echo    -^> Cleaning up executed notebook...
        del "fairness_executed.ipynb"

        cd ..
        echo  -^> Fairness analysis complete.
    )
    if exist "src\shap_analysis.py" (
        echo  -^> Generating SHAP Explainability plots...
        python src/shap_analysis.py
    )
    if exist "src\calibration_analysis.py" (
        echo  -^> Generating Calibration Curves...
        python src/calibration_analysis.py
    )
)
echo.

:: ---------------------------------------------------------
:: 7. Launch Demo App
:: ---------------------------------------------------------
set /p LAUNCH_APP="7. Do you want to launch the demo app? (Y/N by default): "

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
