# Fintech Credit Scoring System 💳📈

A complete end-to-end Machine Learning pipeline and interactive web application for predicting
credit card default risk. This project automates everything from raw data ingestion and
advanced feature engineering to model training, evaluation, fairness analysis, and deployment via Streamlit.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Features & Pipeline Steps](#%EF%B8%8F-features--pipeline-steps)
- [Machine Learning Models](#-machine-learning-models)
- [Fairness Analysis](#%EF%B8%8F-fairness-analysis)
- [Interactive Web App](#%EF%B8%8F-interactive-web-app)

---

## 🎯 Overview

This system uses historical financial behavior (repayment status, bill amounts, past payments)
and demographic data to assess the likelihood of a customer defaulting on their credit card
next month. It features an automated data pipeline that expands a baseline of 18 features
into 44 highly predictive engineered features (e.g., delinquency trends, utilization ratios,
payment pressure). The project also includes a fairness analysis pipeline for evaluating demographic disparities
across different machine learning models.

---

## 📁 Project Structure

```text
├── data/                        
│   ├── processed/              # Processed ML-ready data (3, 18, and 44 features)
│   └── raw/                    # Raw downloaded data (data.xls)
├── models/                     # Trained model checkpoints (.pkl files)
├── notebooks/                  # Jupyter Notebook templates
│   ├── compare.ipynb           # Evaluates model performance using Nested CV
│   ├── eda.ipynb               # Data exploration
│   ├── fairness.ipynb          # Fairness analysis visualizations
│   └── preprocess.ipynb        # Creates train/test splits and saves .pkl
├── results/                    # Evaluation results
│   ├── 3features/              # Results for 3 feature dataset
│   ├── 18features/             # Results for 18 feature dataset
│   ├── 44features/             # Results for 44 feature dataset
│   └── fairness/               # Fairness results
├── src/                        # Core Python modules
│   ├── __init__.py             # Makes src importable
│   ├── data_loader.py          # Used to load .xls files
│   ├── download_data.py        # Fetches raw dataset from UCI repo
│   ├── engineered_features.py  # Generates the 44-feature dataset
│   ├── fairness_analysis.py    # Demographic fairness evaluation
│   ├── preprocessing.py        # Scaling, imputation, and One-Hot Encoding
│   └── *_model.py              # Training scripts for various ML architectures
├── tests/                      # Minimal unit tests
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
└── run.sh / run.bat            # Interactive pipeline automation scripts
```

---

## 🚀 Quickstart Guide

The easiest way to set up and run the project is by using the interactive pipeline manager. It will automatically handle virtual environments, dependencies, data downloads, and execution.

### For Mac/Linux:

1. Open your terminal and navigate to the project directory.
2. Grant execution permissions to the bash script:
```bash
chmod +x run.sh
```


3. Run the interactive setup wizard:
```bash
./run.sh
```



### For Windows:

1. Open Command Prompt or PowerShell in the project directory.
2. Run the batch script:
```cmd
./run.bat
```
*(Follow the on-screen prompts to build your environment, download data, train models, and launch the UI!)*

### Unit tests:
1. Open Command Prompt or PowerShell in the project directory.
2. Run the following line:
```cmd
python -m pytest tests/
```

---

## ⚙️ Features & Pipeline Steps

The pipeline is managed interactively and includes the following capabilities:

1. **Automated Environment Setup:** Seamlessly creates a `.venv` and installs `requirements.txt`.
2. **Data Ingestion:** Downloads the *Default of Credit Card Clients Dataset* directly from the UCI Machine Learning Repository.
3. **Automated Preprocessing:** - Uses `papermill` to execute `notebooks/preprocess.ipynb` safely in the background.
* Cleans the raw data and creates 3-feature, 18-feature, and 44-feature splits.
* Saves robust scikit-learn `ColumnTransformer` pipelines to ensure no data leakage.


4. **Model Training:** Executes scripts in `src/` to train models on the processed data arrays.
5. **Evaluation:** Uses `papermill` to run nested cross-validation and output dynamic evaluation notebooks without overwriting your templates.
6. **Fairness Analysis:** Runs demographic fairness evaluation across SEX, EDUCATION, and MARRIAGE groups and generates fairness visualizations and CSV reports.

---

## 🧠 Machine Learning Models

The system is configured to train and evaluate multiple model architectures to find the best fit for the data:

* **Random Forest** (Default)
* **LightGBM**
* **XGBoost**
* **Gradient Boosting**
* **Logistic Regression**
* **Decision Tree**
* **K-Nearest Neighbors (KNN)**

*Checkpoints are saved dynamically in the `models/` directory matching the format `[model_name]_[n]features.pkl`.*

---

## ⚖️ Fairness Analysis

The project includes a fairness evaluation pipeline implemented in:

```bash
src/fairness_analysis.py
```

and visualized through:

```bash
notebooks/fairness.ipynb
```

The fairness system evaluates demographic subgroup performance across:

* SEX
* EDUCATION
* MARRIAGE

Generated outputs include:

* Approval rate comparisons
* Recall comparisons
* False Negative Rate (FNR) comparisons
* CSV subgroup reports
* Disparity summaries

Results are automatically saved to:

```bash
results/fairness/
```

---

## 🖥️ Interactive Web App

Launch the UI by selecting "Y" to **Launch Demo App** at the end of the `run.sh` script, or run:

```bash
streamlit run app.py

```

**App Features:**

* **Dynamic System Configuration:** Swap between underlying models (e.g., Random Forest vs. LightGBM) and dataset complexities (3, 18, or 44 features) on the fly without restarting the server.
* **Auto-Engineering Engine:** Enter raw customer data and the app instantly calculates the 44 advanced risk metrics in the background before running inference.
* **Adjustable Risk Tolerance:** Use the slider to shift the decision threshold based on business goals (Conservative vs. Aggressive approval policies).
* **Transparency:** View the auto-calculated variables in an expandable dataframe to understand exactly what metrics the model is looking at.
