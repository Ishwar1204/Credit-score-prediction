"""
Streamlit Web Application for Credit Score Prediction.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_dataframe, COMMON_LOAN_TYPES

st.set_page_config(
    page_title="Credit Score AI Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2060df;
    }
    .score-good {
        color: #1b8a5a;
        font-weight: 700;
        font-size: 26px;
    }
    .score-standard {
        color: #d97706;
        font-weight: 700;
        font-size: 26px;
    }
    .score-poor {
        color: #dc2626;
        font-weight: 700;
        font-size: 26px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_package(model_path="models/best_credit_score_model.joblib"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_metrics(metrics_path="models/training_metrics.json"):
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main():
    st.title("💳 AI Credit Score Prediction System")
    st.caption("Machine Learning powered credit risk rating: Poor, Standard, or Good.")

    model_package = load_model_package()
    metrics_data = load_metrics()

    if model_package is None:
        st.warning("⚠️ No trained model found yet! Please train the model first by running `python src/train.py`.")
        return

    model = model_package["model"]
    encoders = model_package["encoders"]
    imputer = model_package.get("imputer", None)
    feature_names = model_package["feature_names"]
    target_inv_map = model_package["target_inv_map"]
    model_name = model_package.get("model_name", "Classifier")

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Customer Assessment",
        "📂 Batch File Prediction",
        "📊 Model Evaluation & Feature Insights"
    ])

    with tab1:
        st.subheader("Customer Financial & Credit Profile")
        st.write("Fill in the customer information below to evaluate creditworthiness.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Personal & Employment")
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            occupation = st.selectbox(
                "Occupation",
                ["Scientist", "Teacher", "Engineer", "Doctor", "Lawyer", "Accountant", 
                 "Architect", "Developer", "Media_Manager", "Journalist", "Entrepreneur", "Unknown"]
            )
            annual_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=500000.0, value=45000.0, step=1000.0)
            monthly_salary = st.number_input("Monthly In-hand Salary ($)", min_value=100.0, max_value=50000.0, value=3500.0, step=100.0)

        with col2:
            st.markdown("##### 🏦 Banking & Liabilities")
            num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=3, step=1)
            num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=4, step=1)
            interest_rate = st.slider("Interest Rate (%)", min_value=1, max_value=40, value=12)
            outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, max_value=200000.0, value=1200.0, step=100.0)
            changed_credit_limit = st.number_input("Changed Credit Limit ($)", min_value=-50.0, max_value=100.0, value=10.0, step=1.0)

        with col3:
            st.markdown("##### 📈 Repayment & Credit Behaviour")
            credit_history_years = st.slider("Credit History Age (Years)", min_value=0, max_value=40, value=10)
            delay_due_days = st.number_input("Average Delay from Due Date (Days)", min_value=0, max_value=60, value=5, step=1)
            num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0, max_value=50, value=4, step=1)
            num_credit_inquiries = st.number_input("Recent Credit Inquiries", min_value=0, max_value=30, value=2, step=1)
            credit_mix = st.selectbox("Credit Mix Quality", ["Good", "Standard", "Bad"])

        st.markdown("---")
        col4, col5 = st.columns(2)

        with col4:
            st.markdown("##### 💳 Spending & Budget")
            credit_utilization = st.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
            total_emi = st.number_input("Total Monthly EMI ($)", min_value=0.0, max_value=20000.0, value=150.0, step=10.0)
            invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, max_value=20000.0, value=120.0, step=10.0)
            monthly_balance = st.number_input("Monthly Ending Balance ($)", min_value=0.0, max_value=50000.0, value=400.0, step=20.0)

        with col5:
            st.markdown("##### 📋 Payment Characteristics & Loans")
            payment_of_min = st.selectbox("Payment of Minimum Amount Only?", ["No", "Yes", "Unknown"])
            payment_behaviour = st.selectbox(
                "Payment Behaviour",
                [
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments",
                    "High_spent_Small_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments",
                    "Unknown"
                ]
            )
            selected_loans = st.multiselect(
                "Active Loan Types",
                COMMON_LOAN_TYPES,
                default=["Auto Loan", "Personal Loan"]
            )

        num_of_loan = len(selected_loans)

        if st.button("🚀 Calculate Credit Score", type="primary", use_container_width=True):
            # Construct single-row DataFrame
            input_dict = {
                "ID": "CUSTOM_01",
                "Customer_ID": "CUS_INPUT",
                "Month": "Current",
                "Name": "Sample User",
                "Age": age,
                "SSN": "000-00-0000",
                "Occupation": occupation,
                "Annual_Income": annual_income,
                "Monthly_Inhand_Salary": monthly_salary,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Card": num_credit_cards,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": num_of_loan,
                "Type_of_Loan": ", ".join(selected_loans),
                "Delay_from_due_date": delay_due_days,
                "Num_of_Delayed_Payment": num_delayed_payments,
                "Changed_Credit_Limit": changed_credit_limit,
                "Num_Credit_Inquiries": num_credit_inquiries,
                "Credit_Mix": credit_mix,
                "Outstanding_Debt": outstanding_debt,
                "Credit_Utilization_Ratio": credit_utilization,
                "Credit_History_Age": f"{credit_history_years} Years and 0 Months",
                "Payment_of_Min_Amount": payment_of_min,
                "Total_EMI_per_month": total_emi,
                "Amount_invested_monthly": invested_monthly,
                "Payment_Behaviour": payment_behaviour,
                "Monthly_Balance": monthly_balance
            }

            raw_df = pd.DataFrame([input_dict])
            clean_df = clean_dataframe(raw_df, is_train=False)

            # Categorical encoding
            for col, le in encoders.items():
                if col in clean_df.columns:
                    val_str = str(clean_df[col].iloc[0])
                    if val_str in le.classes_:
                        clean_df[col] = le.transform([val_str])[0]
                    else:
                        clean_df[col] = le.transform([le.classes_[0]])[0]

            # Reorder / fill features
            for col in feature_names:
                if col not in clean_df.columns:
                    clean_df[col] = np.nan

            X_input = clean_df[feature_names].copy()
            if imputer is not None:
                X_input = imputer.transform(X_input)

            pred_code = model.predict(X_input)[0]
            pred_score = target_inv_map[int(pred_code)]
            probs = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None

            st.markdown("### 🎯 Assessment Results")
            res_c1, res_c2 = st.columns([1, 2])

            with res_c1:
                if pred_score == "Good":
                    st.success("### Rating: **GOOD** 🌟")
                    st.write("🟢 Low risk profile with strong financial stability.")
                elif pred_score == "Standard":
                    st.warning("### Rating: **STANDARD** ⚖️")
                    st.write("🟡 Moderate risk profile with acceptable repayment history.")
                else:
                    st.error("### Rating: **POOR** 🚨")
                    st.write("🔴 High risk profile with elevated debt or frequent payment delays.")

            with res_c2:
                if probs is not None:
                    st.markdown("##### Prediction Confidence")
                    for i, label in target_inv_map.items():
                        prob_pct = float(probs[i])
                        st.progress(prob_pct, text=f"{label}: {prob_pct*100:.1f}%")

            # Financial Health Suggestions
            st.markdown("##### 💡 Key Recommendations to Improve Credit Score:")
            recs = []
            if outstanding_debt > 2000:
                recs.append("🔻 **Reduce Outstanding Debt**: High outstanding debt is one of the heaviest penalties on credit score.")
            if delay_due_days > 7 or num_delayed_payments > 3:
                recs.append("⏰ **Eliminate Payment Delays**: Setting up automatic debit for minimal or full dues prevents severe delinquency penalties.")
            if credit_utilization > 30:
                recs.append("💳 **Lower Credit Utilization**: Keep credit card usage under 30% of total credit limits.")
            if num_credit_inquiries > 4:
                recs.append("🛑 **Limit Hard Credit Inquiries**: Frequent credit card or loan inquiries signal potential distress.")
            if not recs:
                recs.append("🎉 **Excellent habits!** Continue maintaining on-time payments, low debt balance, and steady investment discipline.")

            for r in recs:
                st.markdown(f"- {r}")

    with tab2:
        st.subheader("📂 Batch Predictions on Dataset")
        st.write("Predict credit scores for thousands of customer profiles in seconds.")

        uploaded_file = st.file_uploader("Upload CSV file (or use default test.csv)", type=["csv"])

        if st.button("⚡ Run Batch Prediction", type="primary"):
            with st.spinner("Processing records and running inference..."):
                if uploaded_file is not None:
                    test_df_raw = pd.read_csv(uploaded_file)
                else:
                    test_df_raw = pd.read_csv("test.csv")

                clean_test = clean_dataframe(test_df_raw, is_train=False)

                for col, le in encoders.items():
                    if col in clean_test.columns:
                        known_classes = set(le.classes_)
                        clean_test[col] = clean_test[col].astype(str).apply(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        clean_test[col] = le.transform(clean_test[col])

                for col in feature_names:
                    if col not in clean_test.columns:
                        clean_test[col] = np.nan

                X_batch = clean_test[feature_names].copy()
                if imputer is not None:
                    X_batch = imputer.transform(X_batch)

                batch_preds = model.predict(X_batch)
                batch_probs = model.predict_proba(X_batch) if hasattr(model, "predict_proba") else None

                id_col = test_df_raw["ID"] if "ID" in test_df_raw.columns else pd.Series(range(len(test_df_raw)))
                res_df = pd.DataFrame({
                    "ID": id_col,
                    "Credit_Score": [target_inv_map[int(p)] for p in batch_preds]
                })

                if batch_probs is not None:
                    for i, label in target_inv_map.items():
                        res_df[f"Prob_{label}"] = batch_probs[:, i].round(4)

                st.success(f"Successfully processed {len(res_df):,} records!")
                st.dataframe(res_df.head(100), use_container_width=True)

                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv_bytes,
                    file_name="credit_score_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

    with tab3:
        st.subheader("📊 Model Evaluation & Benchmarks")
        if metrics_data:
            st.markdown(f"**Selected Best Architecture**: `{model_name}`")

            # Comparative table
            summary_table = []
            for m_name, m_stats in metrics_data.items():
                summary_table.append({
                    "Model": m_name,
                    "Validation Accuracy": f"{m_stats['accuracy']*100:.2f}%",
                    "Macro F1-Score": f"{m_stats['macro_f1']:.4f}",
                    "Weighted F1-Score": f"{m_stats['weighted_f1']:.4f}"
                })
            st.table(pd.DataFrame(summary_table))

            if model_name in metrics_data and "top_features" in metrics_data[model_name]:
                st.markdown("##### 🔑 Key Driver Features (Feature Importances)")
                feat_df = pd.DataFrame(
                    list(metrics_data[model_name]["top_features"].items()),
                    columns=["Feature", "Importance Score"]
                ).sort_values(by="Importance Score", ascending=True)

                st.bar_chart(feat_df.set_index("Feature"))

            if model_name in metrics_data and "classification_report" in metrics_data[model_name]:
                st.markdown("##### 📑 Detailed Classification Report")
                cr = metrics_data[model_name]["classification_report"]
                cr_df = pd.DataFrame(cr).transpose().round(4)
                st.dataframe(cr_df, use_container_width=True)
        else:
            st.info("Metrics will display after training completes.")


if __name__ == "__main__":
    main()
