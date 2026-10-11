"""
Interactive Terminal CLI for Credit Score Prediction (Report Edition: Good / Average / Bad | ₹ INR).
"""
import os
import sys
import json
import pandas as pd

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import load_model, predict_single, predict_file, DEFAULT_MODEL_PATH


def prompt_float(prompt_text, default_val):
    val_str = input(f"{prompt_text} [{default_val}]: ").strip()
    if not val_str:
        return float(default_val)
    try:
        return float(val_str)
    except ValueError:
        print(f"Invalid input, using default: {default_val}")
        return float(default_val)


def prompt_int(prompt_text, default_val):
    val_str = input(f"{prompt_text} [{default_val}]: ").strip()
    if not val_str:
        return int(default_val)
    try:
        return int(val_str)
    except ValueError:
        print(f"Invalid input, using default: {default_val}")
        return int(default_val)


def prompt_str(prompt_text, default_val):
    val_str = input(f"{prompt_text} [{default_val}]: ").strip()
    return val_str if val_str else str(default_val)


def interactive_single_customer():
    print("\n--- 📝 Interactive Customer Assessment (Good / Average / Bad | ₹ INR) ---")
    print("Press Enter to accept the default value shown in brackets.\n")

    age = prompt_int("Age", 31)
    occupation = prompt_str("Occupation (Engineer, Doctor, Scientist, Teacher, Lawyer, etc.)", "Engineer")
    annual_income = prompt_float("Annual Income (₹)", 720000.0)
    monthly_salary = prompt_float("Monthly In-hand Salary (₹)", 55000.0)
    num_bank_accounts = prompt_int("Number of Bank Accounts", 3)
    num_credit_cards = prompt_int("Number of Credit Cards", 4)
    interest_rate = prompt_float("Interest Rate (%)", 12.0)
    num_of_loan = prompt_int("Number of Active Loans", 2)
    loan_types = prompt_str("Loan Types (e.g. Auto Loan, Personal Loan)", "Auto Loan, Personal Loan")
    delay_from_due_date = prompt_int("Average Delay from Due Date (days)", 4)
    delayed_payments = prompt_int("Number of Delayed Payments", 3)
    changed_credit_limit = prompt_float("Changed Credit Limit (₹)", 12000.0)
    inquiries = prompt_float("Number of Credit Inquiries", 2.0)
    credit_mix = prompt_str("Credit Mix (Good / Standard / Bad)", "Standard")
    debt = prompt_float("Outstanding Debt (₹)", 45000.0)
    util = prompt_float("Credit Utilization Ratio (%)", 28.5)
    credit_hist = prompt_str("Credit History Age (e.g. '8 Years and 6 Months')", "8 Years and 6 Months")
    payment_min = prompt_str("Payment of Minimum Amount Only? (Yes / No)", "No")
    total_emi = prompt_float("Total Monthly EMI (₹)", 14000.0)
    invested = prompt_float("Amount Invested Monthly (₹)", 10000.0)
    behaviour = prompt_str("Payment Behaviour", "Low_spent_Small_value_payments")
    monthly_balance = prompt_float("Monthly Balance (₹)", 16000.0)

    data = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": loan_types,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": delayed_payments,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": debt,
        "Credit_Utilization_Ratio": util,
        "Credit_History_Age": credit_hist,
        "Payment_of_Min_Amount": payment_min,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invested,
        "Payment_Behaviour": behaviour,
        "Monthly_Balance": monthly_balance
    }

    print("\nEvaluating credit score classification...")
    result = predict_single(data)
    print("\n" + "=" * 60)
    print(f"🎯 PREDICTED CATEGORY: {result['credit_score'].upper()}")
    print(f"Estimated CIBIL Bureau Scale: ~{result.get('cibil_score', 'N/A')} / 900")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("Class Probabilities:")
    for k, v in result['probabilities'].items():
        print(f"  - {k:8s}: {v*100:.2f}%")

    if result['risk_factors']:
        print("\n⚠️ Identified Financial Risk Flags:")
        for rf in result['risk_factors']:
            print(f"  • {rf}")
    else:
        print("\n✅ Healthy Credit Standing! Zero critical risk flags detected.")
    print("=" * 60 + "\n")


def view_model_metrics():
    metrics_path = "models/training_metrics.json"
    if not os.path.exists(metrics_path):
        print("Metrics file not found. Please train models first.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    print("\n================ REPORT MODEL BENCHMARKS (4 ALGORITHMS + LIGHTGBM) ================")
    for model_name, info in data.items():
        print(f"\nModel: {model_name}")
        print(f"  • Accuracy:    {info['accuracy']*100:.2f}%")
        print(f"  • Macro F1:    {info['macro_f1']:.4f}")
        print(f"  • Weighted F1: {info['weighted_f1']:.4f}")

    if "LightGBM" in data and "top_features" in data["LightGBM"]:
        print("\nTop 10 Feature Importances (LightGBM):")
        for feat, score in list(data["LightGBM"]["top_features"].items())[:10]:
            print(f"  - {feat:30s}: {score:.1f}")
    print("===================================================================================\n")


def main():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"⚠️ Model artifact not found at '{DEFAULT_MODEL_PATH}'.")
        print("Please run `python src/train.py` first to train the models.")
        return

    while True:
        print("\n==========================================================")
        print("  💳 CREDIT SCORE PREDICTOR (Ishwar Kumavat ML Report)")
        print("==========================================================")
        print("1. Assess a Single Customer (Interactive Prompt)")
        print("2. Run Sample Customer Demo (Good, Average, Bad)")
        print("3. Batch Predict on CSV (test.csv)")
        print("4. View All 4 Report Model Benchmarks + LightGBM")
        print("5. Exit")
        choice = input("\nSelect an option [1-5]: ").strip()

        if choice == "1":
            interactive_single_customer()
        elif choice == "2":
            from src.predict import main as run_demo_cli
            sys.argv = ["predict.py", "--demo"]
            run_demo_cli()
        elif choice == "3":
            csv_in = input("Input CSV file path [test.csv]: ").strip()
            if not csv_in:
                csv_in = "test.csv"
            csv_out = input("Output CSV file path [test_predictions.csv]: ").strip()
            if not csv_out:
                csv_out = "test_predictions.csv"
            predict_file(csv_in, csv_out)
        elif choice == "4":
            view_model_metrics()
        elif choice == "5" or choice.lower() in ["exit", "q", "quit"]:
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
