"""
Streamlit Web Application Module.

Provides an interactive user interface for the Fintech Credit Scoring System.
Users can input raw financial data, which is automatically engineered into
advanced risk features. The app dynamically loads the selected machine
learning model pipeline to predict credit default risk in real-time.
"""

import os
from typing import Any, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. Base Feature Mapping & True Dataset Medians
# ---------------------------------------------------------
BASE_FEATURES = {
    "X1": {"name": "Credit Limit (LIMIT_BAL)", "default": 140000.0},
    "X5": {"name": "Age", "default": 34},
    "X6": {"name": "Repay Status", "default": 0},
    "X7": {"name": "Repay Status", "default": 0},
    "X8": {"name": "Repay Status", "default": 0},
    "X9": {"name": "Repay Status", "default": 0},
    "X10": {"name": "Repay Status", "default": 0},
    "X11": {
        "name": "Repay Status (-1=duly, 0=paid min, 1=delay for 1 month...)",
        "default": 0,
    },
    "X12": {"name": "Bill Amount", "default": 22382.0},
    "X13": {"name": "Bill Amount", "default": 21200.0},
    "X14": {"name": "Bill Amount", "default": 20089.0},
    "X15": {"name": "Bill Amount", "default": 19052.0},
    "X16": {"name": "Bill Amount", "default": 18105.0},
    "X17": {"name": "Bill Amount", "default": 17071.0},
    "X18": {"name": "Payment Amount", "default": 2100.0},
    "X19": {"name": "Payment Amount", "default": 2009.0},
    "X20": {"name": "Payment Amount", "default": 1800.0},
    "X21": {"name": "Payment Amount", "default": 1500.0},
    "X22": {"name": "Payment Amount", "default": 1500.0},
    "X23": {"name": "Payment Amount", "default": 1500.0},
}

# ---------------------------------------------------------
# 1.5 Engineered Feature Formal Names
# ---------------------------------------------------------
ENGINEERED_FEATURES_MAP = {
    "X24": "Max Delinquency",
    "X25": "Mean Delinquency",
    "X26": "Positive Delq Count",
    "X27": "Severe Delq Count",
    "X28": "Recent Delinquency",
    "X29": "Delinquency Trend",
    "X30": "Ever Severe Delq",
    "X30_0": "Ever Severe Delq (No)",
    "X30_1": "Ever Severe Delq (Yes)",
    "X31": "Bill Mean",
    "X32": "Bill Max",
    "X33": "Bill Std Dev",
    "X34": "Bill Trend",
    "X35": "Pay Mean",
    "X36": "Pay Max",
    "X37": "Pay Std Dev",
    "X38": "Pay Trend",
    "X39": "Zero Pay Count",
    "X40": "Mean Bill Utilization",
    "X41": "Max Bill Utilization",
    "X42": "High Util Count",
    "X43": "Mean Pay Ratio",
    "X44": "Min Pay Ratio",
    "X45": "Underpay Count",
    "X46": "Avg Bill Minus Pay",
    "X47": "Recent Bill Minus Pay",
}


# ---------------------------------------------------------
# 2. Automated Feature Engineering Engine
# ---------------------------------------------------------
def compute_engineered_features(data: Dict[str, float]) -> Dict[str, float]:
    """
    Computes advanced mathematical risk features based on raw UI inputs.

    This acts as the real-time equivalent of the `engineered_features.py` script,
    allowing the web app to expand basic user inputs into the full 44-feature array
    required by the more complex machine learning models.

    Args:
        data (Dict[str, float]): A dictionary mapping feature IDs (e.g., 'X1') to raw numerical inputs.

    Returns:
        Dict[str, float]: An expanded dictionary containing both the original inputs and the calculated features.
    """
    f = data.copy()

    limit = f["X1"]
    pays = [f[f"X{i}"] for i in range(6, 12)]
    bills = [f[f"X{i}"] for i in range(12, 18)]
    payments = [f[f"X{i}"] for i in range(18, 24)]

    f["X24"] = max(pays)
    f["X25"] = float(np.mean(pays))
    f["X26"] = sum(1 for p in pays if p > 0)
    f["X27"] = sum(1 for p in pays if p >= 2)
    f["X28"] = pays[0]
    f["X29"] = pays[0] - pays[-1]

    ever_severe = 1 if f["X27"] > 0 else 0
    f["X30"] = ever_severe
    f["X30_0"] = 1 if ever_severe == 0 else 0
    f["X30_1"] = 1 if ever_severe == 1 else 0

    f["X31"] = float(np.mean(bills))
    f["X32"] = max(bills)
    f["X33"] = float(np.std(bills)) if len(bills) > 1 else 0.0
    f["X34"] = bills[0] - bills[-1]

    f["X35"] = float(np.mean(payments))
    f["X36"] = max(payments)
    f["X37"] = float(np.std(payments)) if len(payments) > 1 else 0.0
    f["X38"] = payments[0] - payments[-1]
    f["X39"] = sum(1 for p in payments if p == 0)

    utils = [b / max(limit, 1) for b in bills]
    f["X40"] = float(np.mean(utils))
    f["X41"] = max(utils)
    f["X42"] = sum(1 for u in utils if u > 0.8)

    pay_ratios = [p / max(b, 1) if b > 0 else 1.0 for p, b in zip(payments, bills)]
    f["X43"] = float(np.mean(pay_ratios))
    f["X44"] = min(pay_ratios)

    f["X45"] = sum(1 for p, b in zip(payments, bills) if p < b)
    diffs = [b - p for b, p in zip(bills, payments)]
    f["X46"] = float(np.mean(diffs))
    f["X47"] = diffs[0]

    return f


# ---------------------------------------------------------
# 3. Dynamic Cache Model Load
# ---------------------------------------------------------
@st.cache_resource
def load_system(
    model_prefix: str, n_features: str
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Dynamically loads the serialized model and its corresponding preprocessor pipeline.

    Utilizes Streamlit's `@st.cache_resource` decorator to cache the models in memory,
    preventing the app from suffering disk I/O lag every time the user clicks "Predict".

    Args:
        model_prefix (str): The filename prefix of the chosen model (e.g., 'random_forest').
        n_features (str): The dataset complexity flag (e.g., '44', '18', or '3').

    Returns:
        Tuple[Optional[Any], Optional[Any]]: A tuple containing the trained model and
            the fitted scikit-learn ColumnTransformer. Returns (None, None) if files are missing.
    """
    model_path = f"models/{model_prefix}_{n_features}features.pkl"
    data_path = f"data/processed/{n_features}features.pkl"

    # Safety check to ensure the files exist before attempting to load
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None

    model = joblib.load(model_path)

    # Unpack the processed data tuple to get the preprocessor (7th element)
    _, _, _, _, _, _, preprocessor = joblib.load(data_path)
    return model, preprocessor


# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")

st.title("Fintech Credit Scoring System")
st.write(
    "Enter the customer's raw behavioral data. Advanced risk features are engineered automatically."
)

input_data: Dict[str, float] = {}

with st.form("customer_form"):

    st.subheader("👤 Customer Profile")
    demo_cols = st.columns(5)
    demo_keys = ["X1", "X5"]

    for idx, x_key in enumerate(demo_keys):
        display_name = BASE_FEATURES[x_key]["name"]
        default_val = BASE_FEATURES[x_key]["default"]

        with demo_cols[idx]:
            if x_key == "X1":
                input_data[x_key] = st.number_input(
                    display_name, value=float(default_val), step=1000.0
                )
            else:
                input_data[x_key] = st.number_input(
                    display_name, value=int(default_val), step=1
                )

    st.markdown("---")

    st.subheader("📅 Financial History (Last 6 Months)")

    months = [
        ("April (Month 1)", ["X11", "X17", "X23"]),
        ("May (Month 2)", ["X10", "X16", "X22"]),
        ("June (Month 3)", ["X9", "X15", "X21"]),
        ("July (Month 4)", ["X8", "X14", "X20"]),
        ("August (Month 5)", ["X7", "X13", "X19"]),
        ("September (Month 6)", ["X6", "X12", "X18"]),
    ]

    for month_name, keys in months:
        st.markdown(f"**{month_name}**")
        fin_cols = st.columns(3)

        for idx, x_key in enumerate(keys):
            display_name = BASE_FEATURES[x_key]["name"]
            default_val = BASE_FEATURES[x_key]["default"]

            with fin_cols[idx]:
                if idx == 0:
                    input_data[x_key] = st.number_input(
                        f"{display_name} ({month_name.split(' ')[0]})",
                        value=int(default_val),
                        step=1,
                        key=x_key,
                    )
                else:
                    input_data[x_key] = st.number_input(
                        f"{display_name} ({month_name.split(' ')[0]})",
                        value=float(default_val),
                        step=100.0,
                        key=x_key,
                    )

    st.markdown("---")

    st.subheader("⚙️ System & Risk Configuration")
    st.write(
        "Select the machine learning architecture, dataset complexity, and your business risk tolerance."
    )

    # Changed from 2 columns to 3 columns
    sys_cols = st.columns(3)

    model_options = {
        "Random Forest": "random_forest",
        "LightGBM": "lightgbm",
        "XGBoost": "xgboost",
        "Gradient Boosting": "gradient_boosting",
        "Logistic Regression": "logreg",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
    }

    feature_options = {
        "44 Features (Full Engineered)": "44",
        "18 Features (Baseline)": "18",
        "3 Features (Minimal)": "3",
    }

    with sys_cols[0]:
        selected_model_display = st.selectbox(
            "Model Architecture", list(model_options.keys()), index=0
        )

    with sys_cols[1]:
        selected_feature_display = st.selectbox(
            "Dataset", list(feature_options.keys()), index=0
        )

    with sys_cols[2]:
        # Tucked the slider into the third column
        decision_threshold = st.slider(
            "Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
            help="Models output a probability between 0 and 1. Lower values catch more defaults but trigger more false alarms.",
        )

    submit = st.form_submit_button(
        "Predict Default Risk", type="primary", use_container_width=True
    )

# ---------------------------------------------------------
# 5. Feature Computation & Inference
# ---------------------------------------------------------
if submit:
    # Load the specific model requested by the user just in time
    model_code = model_options[selected_model_display]
    feature_code = feature_options[selected_feature_display]

    model, preprocessor = load_system(model_code, feature_code)

    if model is None:
        st.error(
            f"⚠️ Missing Pipeline Artifacts: Could not find trained files for **{selected_model_display}** using the **{feature_code}-feature** dataset. Please run the model training scripts for this combination first."
        )
    else:
        full_customer_data = compute_engineered_features(input_data)
        input_df = pd.DataFrame([full_customer_data])

        try:
            if hasattr(preprocessor, "feature_names_in_"):
                prep_expected_features = list(preprocessor.feature_names_in_)
                for col in prep_expected_features:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                # Slices down the 44 generated features to the exact number required by the loaded preprocessor
                input_df = input_df[prep_expected_features]

            processed = preprocessor.transform(input_df)

            if hasattr(model, "feature_names_in_"):
                processed = pd.DataFrame(processed, columns=model.feature_names_in_)

            prob = model.predict_proba(processed)[0][1]

            st.subheader("Risk Assessment")

            if decision_threshold < 0.40:
                st.info(
                    "📉 Executed under **Conservative Policy**: Prioritizing default capture over approval volume."
                )
            elif decision_threshold > 0.60:
                st.info(
                    "📈 Executed under **Aggressive Policy**: Prioritizing approval volume over default capture."
                )

            warning_threshold = decision_threshold / 2.0

            if prob >= decision_threshold:
                st.error(f"🔴 High Risk (Default Probability: {prob:.2%})")
                st.write(
                    f"This exceeds the current strictness threshold of **{decision_threshold:.2%}**."
                )
            elif prob >= warning_threshold:
                st.warning(f"🟡 Medium Risk (Default Probability: {prob:.2%})")
                st.write(
                    f"Customer is approaching the decision threshold. Manual review recommended."
                )
            else:
                st.success(f"🟢 Low Risk (Default Probability: {prob:.2%})")

            with st.expander("View Auto-Engineered Risk Variables"):
                # Ensure we only display features that were actually used by the model pipeline
                used_cols = [
                    c
                    for c in input_df.columns
                    if c.startswith("X") and int(c[1:].split("_")[0]) >= 24
                ]
                engineered_df = input_df[used_cols]
                display_df = engineered_df.rename(columns=ENGINEERED_FEATURES_MAP)
                display_df = display_df.T.rename(columns={0: "Calculated Value"})
                st.dataframe(display_df, width="stretch")  # type: ignore

        except Exception as e:
            st.error(f"Error processing inputs: {str(e)}")
