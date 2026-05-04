

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

OUTPUT_FILE = OUTPUT_DIR / "engineered_features.xlsx"


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
def load_raw_data(file_path: Path) -> pd.DataFrame:
    # For .xls you may need: pip install xlrd
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


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

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
    util_df = pd.DataFrame({
        f"util_{i}": out[f"BILL_AMT{i}"] / (out["LIMIT_BAL"] + eps)
        for i in range(1, 7)
    })
    
    ratio_df = pd.DataFrame({
        f"pay_ratio_{i}": out[f"PAY_AMT{i}"] / (out[f"BILL_AMT{i}"].abs() + eps)
        for i in range(1, 7)
    })


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


# ============================================================
# 3. SAVE OUTPUT
# ============================================================
def save_output(df: pd.DataFrame, output_file: Path):
    df.to_excel(output_file, index=False)
    print(f"Engineered dataset saved to: {output_file}")


# ============================================================
# 4. MAIN
# ============================================================
def main():
    df_raw = load_raw_data(RAW_FILE)
    print("Raw dataset shape:", df_raw.shape)

    df_engineered = engineer_features(df_raw)
    print("Engineered dataset shape:", df_engineered.shape)

    save_output(df_engineered, OUTPUT_FILE)

    print("\n All engineered columns added:")
    original_cols = set(df_raw.columns)
    new_cols = [col for col in df_engineered.columns if col not in original_cols]
    print(new_cols[:24])

    print("\nTotal new engineered features:", len(new_cols))


if __name__ == "__main__":
    main()