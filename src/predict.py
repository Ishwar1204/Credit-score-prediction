"""
Batch and single customer prediction module for Credit Score Classification.
Based on Ishwar Kumavat's ML Report (Target: Good, Average, Bad | Currency: ₹ INR).
"""
import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_dataframe, COMMON_LOAN_TYPES

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "best_credit_score_model.joblib"
)


def load_model(model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run training first.")
    return joblib.load(model_path)


def predict_single(data_dict: dict, model_artifacts=None):
    """
    Predict credit score for a single customer profile dictionary.
    Returns:
        dict: {
            'credit_score': 'Good' | 'Average' | 'Bad',
            'probabilities': {'Bad': float, 'Average': float, 'Good': float},
            'risk_factors': list of str,
            'confidence': float,
            'cibil_score': int
        }
    """
    if model_artifacts is None:
        model_artifacts = load_model()

    model = model_artifacts["model"]
    imputer = model_artifacts.get("imputer", None)
    scaler = model_artifacts.get("scaler", None)
    encoders = model_artifacts["encoders"]
    feature_names = model_artifacts["feature_names"]
    target_inv_map = model_artifacts.get("target_inv_map", {0: 'Bad', 1: 'Average', 2: 'Good'})

    raw_df = pd.DataFrame([data_dict])
    clean_df = clean_dataframe(raw_df, is_train=False)

    for col, le in encoders.items():
        if col in clean_df.columns:
            val_str = str(clean_df[col].iloc[0])
            if val_str in le.classes_:
                clean_df[col] = le.transform([val_str])[0]
            else:
                clean_df[col] = le.transform([le.classes_[0]])[0]

    for col in feature_names:
        if col not in clean_df.columns:
            clean_df[col] = np.nan

    X = clean_df[feature_names].copy()
    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)

    raw_pred = model.predict(X)
    pred_code = int(np.asarray(raw_pred).ravel()[0])
    pred_label = target_inv_map[pred_code]

    probs = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(X)[0]
        for idx, lbl in target_inv_map.items():
            probs[lbl] = round(float(prob_arr[idx]), 4)
        confidence = probs[pred_label]

    p_good = probs.get("Good", 0.0)
    p_avg = probs.get("Average", 0.0)
    p_bad = probs.get("Bad", 0.0)
    cibil_est = int(350 + (p_bad * 160) + (p_avg * 310) + (p_good * 470))
    cibil_est = max(300, min(890, cibil_est))

    risk_factors = []
    debt = data_dict.get("Outstanding_Debt", 0)
    delay_days = data_dict.get("Delay_from_due_date", 0)
    delayed_payments = data_dict.get("Num_of_Delayed_Payment", 0)
    util = data_dict.get("Credit_Utilization_Ratio", 0)
    inquiries = data_dict.get("Num_Credit_Inquiries", 0)
    monthly_sal = data_dict.get("Monthly_Inhand_Salary", 0)
    emi = data_dict.get("Total_EMI_per_month", 0)

    try:
        debt = float(debt)
    except (ValueError, TypeError):
        debt = 0
    try:
        delay_days = float(delay_days)
    except (ValueError, TypeError):
        delay_days = 0
    try:
        delayed_payments = float(delayed_payments)
    except (ValueError, TypeError):
        delayed_payments = 0
    try:
        util = float(util)
    except (ValueError, TypeError):
        util = 0
    try:
        inquiries = float(inquiries)
    except (ValueError, TypeError):
        inquiries = 0
    try:
        monthly_sal = float(monthly_sal)
    except (ValueError, TypeError):
        monthly_sal = 0
    try:
        emi = float(emi)
    except (ValueError, TypeError):
        emi = 0

    if debt > 50000:
        risk_factors.append(f"High Outstanding Debt (₹{debt:,.0f})")
    elif debt > 2500 and monthly_sal > 0 and (debt / monthly_sal) > 1.5:
        risk_factors.append(f"Elevated Debt Burden (₹{debt:,.0f} outstanding)")

    if monthly_sal > 0 and (emi / monthly_sal) > 0.40:
        emi_pct = (emi / monthly_sal) * 100
        risk_factors.append(f"Heavy EMI Obligation ({emi_pct:.1f}% of in-hand salary; recommended < 40%)")

    if delay_days > 7:
        risk_factors.append(f"Delinquency Flag ({delay_days} days past payment due date)")

    if delayed_payments > 3:
        risk_factors.append(f"Frequent Late Payments ({delayed_payments} recorded delayed payments)")

    if util > 30:
        risk_factors.append(f"High Credit Card Utilization ({util:.1f}%; ideal threshold < 30%)")

    if inquiries > 4:
        risk_factors.append(f"Multiple Bureau Inquiries ({inquiries} recent hard checks)")

    return {
        "credit_score": pred_label,
        "confidence": confidence,
        "probabilities": probs,
        "risk_factors": risk_factors,
        "cibil_score": cibil_est
    }


def predict_file(
    input_path="test.csv",
    output_path="test_predictions.csv",
    model_path=DEFAULT_MODEL_PATH
):
    """
    Run predictions on a CSV file and output results with probabilities to output_path.
    """
    artifacts = load_model(model_path)
    model = artifacts["model"]
    imputer = artifacts.get("imputer", None)
    scaler = artifacts.get("scaler", None)
    encoders = artifacts["encoders"]
    feature_names = artifacts["feature_names"]
    target_inv_map = artifacts.get("target_inv_map", {0: 'Bad', 1: 'Average', 2: 'Good'})

    print(f"--> Reading input data from '{input_path}'...")
    df_raw = pd.read_csv(input_path, low_memory=False)
    id_series = df_raw["ID"] if "ID" in df_raw.columns else pd.Series(range(len(df_raw)))

    print(f"--> Preprocessing {len(df_raw):,} records with 46 financial features...")
    df_clean = clean_dataframe(df_raw, is_train=False)

    for col, le in encoders.items():
        if col in df_clean.columns:
            known_classes = set(le.classes_)
            series_clean = df_clean[col].astype(str)
            series_clean = series_clean.apply(lambda x: x if x in known_classes else le.classes_[0])
            df_clean[col] = le.transform(series_clean)

    for col in feature_names:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    X = df_clean[feature_names].copy()
    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)

    print("--> Generating predictions (Good, Average, Bad)...")
    preds_numeric = np.asarray(model.predict(X)).ravel().astype(int)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    pred_labels = [target_inv_map[int(p)] for p in preds_numeric]

    out_df = pd.DataFrame({
        "ID": id_series,
        "Credit_Score": pred_labels
    })

    if probs is not None:
        for idx, label in target_inv_map.items():
            out_df[f"Prob_{label}"] = probs[:, idx].round(4)

    out_df.to_csv(output_path, index=False)
    print(f"--> Successfully saved predictions to: '{output_path}'")
    print("\nClass Distribution:")
    print(out_df["Credit_Score"].value_counts().to_string())

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Credit Score Prediction CLI (Good, Average, Bad | ₹ INR)")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV file to predict on")
    parser.add_argument("--output", type=str, default="test_predictions.csv", help="Path to save prediction output CSV")
    parser.add_argument("--demo", action="store_true", help="Run a quick prediction demonstration on sample customer profiles")
    args = parser.parse_args()

    if args.demo:
        print("\n================ CREDIT SCORE PREDICTION DEMO (Good / Average / Bad) ================\n")
        sample_customers = [
            {
                "name": "Profile 1: High Earner, Low Debt, On-time Payer (Expected: Good)",
                "data": {
                    "Age": 38,
                    "Occupation": "Doctor",
                    "Annual_Income": 1500000.0,
                    "Monthly_Inhand_Salary": 110000.0,
                    "Num_Bank_Accounts": 2,
                    "Num_Credit_Card": 3,
                    "Interest_Rate": 7.5,
                    "Num_of_Loan": 1,
                    "Type_of_Loan": "Home Equity Loan",
                    "Delay_from_due_date": 1,
                    "Num_of_Delayed_Payment": 0,
                    "Changed_Credit_Limit": 5000.0,
                    "Num_Credit_Inquiries": 1,
                    "Credit_Mix": "Good",
                    "Outstanding_Debt": 25000.0,
                    "Credit_Utilization_Ratio": 15.0,
                    "Credit_History_Age": "18 Years and 4 Months",
                    "Payment_of_Min_Amount": "No",
                    "Total_EMI_per_month": 12000.0,
                    "Amount_invested_monthly": 35000.0,
                    "Payment_Behaviour": "High_spent_Small_value_payments",
                    "Monthly_Balance": 45000.0
                }
            },
            {
                "name": "Profile 2: Moderate Income, Steady Installments (Expected: Average)",
                "data": {
                    "Age": 31,
                    "Occupation": "Engineer",
                    "Annual_Income": 720000.0,
                    "Monthly_Inhand_Salary": 55000.0,
                    "Num_Bank_Accounts": 3,
                    "Num_Credit_Card": 4,
                    "Interest_Rate": 12.0,
                    "Num_of_Loan": 2,
                    "Type_of_Loan": "Auto Loan, Personal Loan",
                    "Delay_from_due_date": 4,
                    "Num_of_Delayed_Payment": 3,
                    "Changed_Credit_Limit": 12000.0,
                    "Num_Credit_Inquiries": 2,
                    "Credit_Mix": "Standard",
                    "Outstanding_Debt": 45000.0,
                    "Credit_Utilization_Ratio": 28.5,
                    "Credit_History_Age": "8 Years and 6 Months",
                    "Payment_of_Min_Amount": "No",
                    "Total_EMI_per_month": 14000.0,
                    "Amount_invested_monthly": 10000.0,
                    "Payment_Behaviour": "Low_spent_Small_value_payments",
                    "Monthly_Balance": 16000.0
                }
            },
            {
                "name": "Profile 3: Delays, High Debt & Poor Mix (Expected: Bad)",
                "data": {
                    "Age": 24,
                    "Occupation": "Teacher",
                    "Annual_Income": 300000.0,
                    "Monthly_Inhand_Salary": 23000.0,
                    "Num_Bank_Accounts": 7,
                    "Num_Credit_Card": 8,
                    "Interest_Rate": 25,
                    "Num_of_Loan": 5,
                    "Type_of_Loan": "Payday Loan, Personal Loan",
                    "Delay_from_due_date": 24,
                    "Num_of_Delayed_Payment": 15,
                    "Changed_Credit_Limit": 5000.0,
                    "Num_Credit_Inquiries": 8,
                    "Credit_Mix": "Bad",
                    "Outstanding_Debt": 85000.0,
                    "Credit_Utilization_Ratio": 48.0,
                    "Credit_History_Age": "1 Years and 6 Months",
                    "Payment_of_Min_Amount": "Yes",
                    "Total_EMI_per_month": 12000.0,
                    "Amount_invested_monthly": 1000.0,
                    "Payment_Behaviour": "Low_spent_Small_value_payments",
                    "Monthly_Balance": 2000.0
                }
            }
        ]

        artifacts = load_model()
        for sample in sample_customers:
            print(f"👤 {sample['name']}")
            res = predict_single(sample["data"], artifacts)
            print(f"   ▶ Result: Category = {res['credit_score'].upper()} (CIBIL: ~{res['cibil_score']}, Conf: {res['confidence']*100:.1f}%)")
            print(f"   ▶ Probabilities: {res['probabilities']}")
            if res["risk_factors"]:
                print(f"   ⚠️ Risk Factors: {', '.join(res['risk_factors'])}")
            print("-" * 75)

    elif args.file:
        predict_file(args.file, args.output)
    else:
        print("Running batch prediction on default 'test.csv'...")
        predict_file("test.csv", args.output)


if __name__ == "__main__":
    main()
