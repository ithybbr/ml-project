"""
Feature Engineering Module.

This script processes the raw financial dataset to construct the 44-feature dataset.
It mathematically derives advanced risk indicators including delinquency trends,
credit utilization ratios, and payment behavior metrics. It also manages the complex
double-header serialization required for scikit-learn compatibility.
"""

from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "data.xls"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "44features.xls"

# ============================================================
# COLUMN MAPPING
# ============================================================
FEATURE_MAP = {
    "ID": "",  # Keeps the cell directly above 'ID' blank, matching your image
    "LIMIT_BAL": "X1",
    "SEX": "X2",
    "EDUCATION": "X3",
    "MARRIAGE": "X4",
    "AGE": "X5",
    "PAY_0": "X6",
    "PAY_2": "X7",
    "PAY_3": "X8",
    "PAY_4": "X9",
    "PAY_5": "X10",
    "PAY_6": "X11",
    "BILL_AMT1": "X12",
    "BILL_AMT2": "X13",
    "BILL_AMT3": "X14",
    "BILL_AMT4": "X15",
    "BILL_AMT5": "X16",
    "BILL_AMT6": "X17",
    "PAY_AMT1": "X18",
    "PAY_AMT2": "X19",
    "PAY_AMT3": "X20",
    "PAY_AMT4": "X21",
    "PAY_AMT5": "X22",
    "PAY_AMT6": "X23",
    "delq_max": "X24",
    "delq_mean": "X25",
    "delq_count_positive": "X26",
    "delq_count_severe": "X27",
    "delq_recent": "X28",
    "delq_trend": "X29",
    "ever_severe_delq": "X30",
    "bill_mean": "X31",
    "bill_max": "X32",
    "bill_std": "X33",
    "bill_trend": "X34",
    "pay_mean": "X35",
    "pay_max": "X36",
    "pay_std": "X37",
    "pay_trend": "X38",
    "zero_pay_count": "X39",
    "bill_utilization_mean": "X40",
    "bill_utilization_max": "X41",
    "high_util_count": "X42",
    "pay_ratio_mean": "X43",
    "pay_ratio_min": "X44",
    "underpay_count": "X45",
    "avg_bill_minus_pay": "X46",
    "recent_bill_minus_pay": "X47",
    "DEFAULT": "Y",
}


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Ingests the raw Excel file and standardizes column formats.

    Resolves edge cases such as shifted headers present in the original dataset
    and enforces correct datatypes prior to transformation.

    Args:
        file_path (Path): The path to the raw dataset on disk.

    Returns:
        pd.DataFrame: A cleaned dataframe with unified column names.
    """
    df = pd.read_excel(file_path)

    # In this dataset, the first row often contains the real column names
    first_row = df.iloc[0].astype(str).tolist()
    if "LIMIT_BAL" in first_row or "default payment next month" in first_row:
        df.columns = first_row
        df = df.iloc[1:].copy()

    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "DEFAULT"})

    if df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})

    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with target
    df = df.dropna(subset=["DEFAULT"]).copy()
    df["DEFAULT"] = df["DEFAULT"].astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced predictive features based on historical financial tracking.

    Derives secondary statistics such as maximum delinquency, utilization percentages,
    payment trends, and debt ratios.

    Args:
        df (pd.DataFrame): The base dataframe containing standard raw features.

    Returns:
        pd.DataFrame: A horizontally expanded dataframe containing the original data
            plus the newly engineered columns.
    """
    out = df.copy()

    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]
    pay_amt_cols = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    eps = 1.0

    # ----------------------------
    # Delinquency features
    # ----------------------------
    out["delq_max"] = out[pay_status_cols].max(axis=1)
    out["delq_mean"] = out[pay_status_cols].mean(axis=1)
    out["delq_count_positive"] = (out[pay_status_cols] > 0).sum(axis=1)
    out["delq_count_severe"] = (out[pay_status_cols] >= 2).sum(axis=1)
    out["delq_recent"] = out["PAY_0"]
    out["delq_trend"] = out["PAY_0"] - out["PAY_6"]
    out["ever_severe_delq"] = (out["delq_max"] >= 2).astype(int)

    # ----------------------------
    # Bill features
    # ----------------------------
    out["bill_mean"] = out[bill_cols].mean(axis=1)
    out["bill_max"] = out[bill_cols].max(axis=1)
    out["bill_std"] = out[bill_cols].std(axis=1).fillna(0)
    out["bill_trend"] = out["BILL_AMT1"] - out["BILL_AMT6"]

    # ----------------------------
    # Payment features
    # ----------------------------
    out["pay_mean"] = out[pay_amt_cols].mean(axis=1)
    out["pay_max"] = out[pay_amt_cols].max(axis=1)
    out["pay_std"] = out[pay_amt_cols].std(axis=1).fillna(0)
    out["pay_trend"] = out["PAY_AMT1"] - out["PAY_AMT6"]
    out["zero_pay_count"] = (out[pay_amt_cols] == 0).sum(axis=1)

    # ----------------------------
    # Utilization + payment ratio
    # ----------------------------
    util_df = pd.DataFrame(
        {
            f"util_{i}": out[f"BILL_AMT{i}"] / (out["LIMIT_BAL"] + eps)
            for i in range(1, 7)
        }
    )

    ratio_df = pd.DataFrame(
        {
            f"pay_ratio_{i}": out[f"PAY_AMT{i}"] / (out[f"BILL_AMT{i}"].abs() + eps)
            for i in range(1, 7)
        }
    )

    out["bill_utilization_mean"] = util_df.mean(axis=1)
    out["bill_utilization_max"] = util_df.max(axis=1)
    out["high_util_count"] = (util_df > 0.8).sum(axis=1)

    out["pay_ratio_mean"] = ratio_df.mean(axis=1)
    out["pay_ratio_min"] = ratio_df.min(axis=1)
    out["underpay_count"] = (ratio_df < 0.2).sum(axis=1)

    # ----------------------------
    # Pressure features
    # ----------------------------
    out["avg_bill_minus_pay"] = out["bill_mean"] - out["pay_mean"]
    out["recent_bill_minus_pay"] = out["BILL_AMT1"] - out["PAY_AMT1"]

    return out


def save_output(df: pd.DataFrame, output_file: Path) -> None:
    """
    Serializes the dataframe to an Excel file formatted with a dual header row.

    The first row contains the generic 'X' variable mappings (e.g., X1, X2), and the
    second row contains the formal textual names.

    Args:
        df (pd.DataFrame): The fully engineered dataframe.
        output_file (Path): The target save location on disk.
    """
    # 1. Ensure columns are ordered sequentially based on the map
    ordered_formal_cols = [col for col in FEATURE_MAP.keys() if col in df.columns]
    df_ordered = df[ordered_formal_cols]

    # 2. Extract the "X" names for the top row
    x_columns = [FEATURE_MAP[col] for col in df_ordered.columns]

    # 3. Create a 1-row DataFrame containing the formal names
    second_header_row = pd.DataFrame([df_ordered.columns.values], columns=x_columns)

    # 4. Rename the main dataframe's columns to the "X" names
    df_ordered.columns = x_columns

    # 5. Stack the formal names row on top of the actual data
    final_df = pd.concat([second_header_row, df_ordered], ignore_index=True)

    # 6. Save to Excel safely
    final_df.to_excel(output_file, index=False)
    print(f"Engineered dataset saved to: {output_file}")


def main() -> None:
    """
    Executes the ingestion, engineering, and serialization sequence.
    """
    df_raw = load_raw_data(RAW_FILE)
    print("Raw dataset shape:", df_raw.shape)

    df_engineered = engineer_features(df_raw)
    print("Engineered dataset shape:", df_engineered.shape)

    save_output(df_engineered, OUTPUT_FILE)

    print("\nAll engineered columns formatted with double-headers successfully.")


if __name__ == "__main__":
    main()
