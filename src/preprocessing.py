"""
Enhanced Data Preprocessing & Feature Engineering for Credit Score Classification.
Engineered with Indian Rupee (₹) financial context and high-predictive domain ratios.
"""
import re
import numpy as np
import pandas as pd


def parse_credit_history_age(val):
    """
    Parses strings like '22 Years and 1 Months' into total months.
    Returns np.nan if unparseable.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r'(\d+)\s*Years?\s*(?:and)?\s*(\d+)\s*Months?', val_str, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    match_single = re.search(r'(\d+)', val_str)
    if match_single:
        return float(match_single.group(1)) * 12
    return np.nan


def clean_numeric_series(series):
    """
    Cleans series with trailing underscores, weird characters, or dirty strings.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r'[^0-9\.\-]', '', regex=True)
        .str.strip()
    )
    cleaned = cleaned.replace('', np.nan)
    return pd.to_numeric(cleaned, errors='coerce')


COMMON_LOAN_TYPES = [
    'Auto Loan',
    'Credit-Builder Loan',
    'Personal Loan',
    'Home Equity Loan',
    'Mortgage Loan',
    'Student Loan',
    'Debt Consolidation Loan',
    'Payday Loan'
]


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Parses dirty numerical strings (Age, Annual_Income, Outstanding_Debt, etc.)
    - Formats Credit_History_Age into total months
    - Handles placeholders in categoricals ('_______', '!@9#%12', '_')
    - Extracts binary indicators for loan types
    - Computes advanced financial ratios:
        * Debt to Income ratios
        * EMI to Salary ratio
        * Savings & Discretionary Cash ratios
        * Delay Severity index
        * Inquiries per Account
        * Total accounts & loans per account
    """
    data = df.copy()

    # Clean numeric columns with string noise
    dirty_numeric_cols = [
        'Age',
        'Annual_Income',
        'Num_of_Loan',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Outstanding_Debt',
        'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    for col in dirty_numeric_cols:
        if col in data.columns:
            data[col] = clean_numeric_series(data[col])

    # Standard numeric columns conversion
    standard_num_cols = [
        'Monthly_Inhand_Salary',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Delay_from_due_date',
        'Num_Credit_Inquiries',
        'Credit_Utilization_Ratio',
        'Total_EMI_per_month'
    ]
    for col in standard_num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Convert Credit_History_Age to months
    if 'Credit_History_Age' in data.columns:
        data['Credit_History_Months'] = data['Credit_History_Age'].apply(parse_credit_history_age)
        data.drop(columns=['Credit_History_Age'], inplace=True)

    # Sanity clipping and outlier treatment
    if 'Age' in data.columns:
        data.loc[(data['Age'] < 14) | (data['Age'] > 100), 'Age'] = np.nan

    if 'Num_Bank_Accounts' in data.columns:
        data.loc[(data['Num_Bank_Accounts'] < 0) | (data['Num_Bank_Accounts'] > 50), 'Num_Bank_Accounts'] = np.nan

    if 'Num_Credit_Card' in data.columns:
        data.loc[(data['Num_Credit_Card'] < 0) | (data['Num_Credit_Card'] > 50), 'Num_Credit_Card'] = np.nan

    if 'Interest_Rate' in data.columns:
        data.loc[(data['Interest_Rate'] < 0) | (data['Interest_Rate'] > 100), 'Interest_Rate'] = np.nan

    if 'Num_of_Loan' in data.columns:
        data.loc[(data['Num_of_Loan'] < 0) | (data['Num_of_Loan'] > 30), 'Num_of_Loan'] = np.nan

    if 'Delay_from_due_date' in data.columns:
        data.loc[data['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = 0

    if 'Num_of_Delayed_Payment' in data.columns:
        data.loc[(data['Num_of_Delayed_Payment'] < 0) | (data['Num_of_Delayed_Payment'] > 100), 'Num_of_Delayed_Payment'] = np.nan

    if 'Num_Credit_Inquiries' in data.columns:
        data.loc[(data['Num_Credit_Inquiries'] < 0) | (data['Num_Credit_Inquiries'] > 100), 'Num_Credit_Inquiries'] = np.nan

    if 'Outstanding_Debt' in data.columns:
        data.loc[data['Outstanding_Debt'] < 0, 'Outstanding_Debt'] = np.nan

    if 'Annual_Income' in data.columns:
        data.loc[data['Annual_Income'] <= 0, 'Annual_Income'] = np.nan

    # Fill missing Monthly_Inhand_Salary approximately if Annual_Income is present
    if 'Monthly_Inhand_Salary' in data.columns and 'Annual_Income' in data.columns:
        salary_est = data['Annual_Income'] / 12.0
        data['Monthly_Inhand_Salary'] = data['Monthly_Inhand_Salary'].fillna(salary_est)

    # Clean Loan Types and extract multi-hot binary indicators
    total_detected_loans = pd.Series(0, index=data.index)
    if 'Type_of_Loan' in data.columns:
        loan_series = data['Type_of_Loan'].fillna('').astype(str)
        for loan_t in COMMON_LOAN_TYPES:
            col_name = f"Loan_{loan_t.replace(' ', '_').replace('-', '_')}"
            has_loan = loan_series.str.contains(loan_t, case=False, regex=False).astype(int)
            data[col_name] = has_loan
            total_detected_loans += has_loan
        data.drop(columns=['Type_of_Loan'], inplace=True)

    data['Detected_Loans_Count'] = total_detected_loans

    # Clean Categoricals
    if 'Occupation' in data.columns:
        data['Occupation'] = data['Occupation'].astype(str).str.strip()
        data['Occupation'] = data['Occupation'].replace(['_______', '', 'nan'], 'Unknown')

    if 'Credit_Mix' in data.columns:
        data['Credit_Mix'] = data['Credit_Mix'].astype(str).str.strip()
        data['Credit_Mix'] = data['Credit_Mix'].replace(['_', '', 'nan'], 'Standard')
        mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
        data['Credit_Mix_Score'] = data['Credit_Mix'].map(mix_map).fillna(1).astype(int)

    if 'Payment_of_Min_Amount' in data.columns:
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].astype(str).str.strip()
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].replace(['NM', '', 'nan'], 'Unknown')
        min_map = {'No': 2, 'Unknown': 1, 'Yes': 0}
        data['Payment_Min_Score'] = data['Payment_of_Min_Amount'].map(min_map).fillna(1).astype(int)

    if 'Payment_Behaviour' in data.columns:
        data['Payment_Behaviour'] = data['Payment_Behaviour'].astype(str).str.strip()
        data['Payment_Behaviour'] = data['Payment_Behaviour'].replace(['!@9#%12', '', 'nan'], 'Unknown')

    # Advanced Domain Financial Ratios
    salary = data['Monthly_Inhand_Salary'].replace(0, np.nan)
    annual = data['Annual_Income'].replace(0, np.nan)

    # 1. Debt-to-Income Ratios
    if 'Outstanding_Debt' in data.columns:
        data['Debt_to_Annual_Income'] = data['Outstanding_Debt'] / (annual + 1e-4)
        data['Debt_to_Monthly_Salary'] = data['Outstanding_Debt'] / (salary + 1e-4)

    # 2. EMI & Installment Burden Ratios
    if 'Total_EMI_per_month' in data.columns:
        data['EMI_to_Salary_Ratio'] = data['Total_EMI_per_month'] / (salary + 1e-4)

    # 3. Monthly Savings & Balance Ratios
    if 'Monthly_Balance' in data.columns:
        data['Balance_to_Salary_Ratio'] = data['Monthly_Balance'] / (salary + 1e-4)
        if 'Outstanding_Debt' in data.columns:
            data['Balance_to_Debt_Ratio'] = data['Monthly_Balance'] / (data['Outstanding_Debt'] + 1.0)

    # 4. Investment Propensity Ratio
    if 'Amount_invested_monthly' in data.columns:
        data['Invest_to_Salary_Ratio'] = data['Amount_invested_monthly'] / (salary + 1e-4)

    # 5. Net Discretionary Cash Flow
    if 'Total_EMI_per_month' in data.columns and 'Amount_invested_monthly' in data.columns:
        emi = data['Total_EMI_per_month'].fillna(0)
        inv = data['Amount_invested_monthly'].fillna(0)
        data['Discretionary_Income'] = data['Monthly_Inhand_Salary'] - emi - inv
        data['Discretionary_Ratio'] = data['Discretionary_Income'] / (salary + 1e-4)

    # 6. Accounts & Utilization Intensity
    num_cards = data.get('Num_Credit_Card', pd.Series(0, index=data.index)).fillna(0)
    num_banks = data.get('Num_Bank_Accounts', pd.Series(0, index=data.index)).fillna(0)
    data['Total_Accounts'] = num_banks + num_cards

    if 'Num_Credit_Inquiries' in data.columns:
        data['Inquiries_per_Card'] = data['Num_Credit_Inquiries'] / (num_cards + 1.0)
        data['Inquiries_per_Account'] = data['Num_Credit_Inquiries'] / (data['Total_Accounts'] + 1.0)

    # 7. Delay Severity Score
    if 'Delay_from_due_date' in data.columns and 'Num_of_Delayed_Payment' in data.columns:
        data['Delay_Severity_Index'] = data['Delay_from_due_date'] * data['Num_of_Delayed_Payment']

    # 8. Delinquency frequency over history
    if 'Credit_History_Months' in data.columns and 'Num_of_Delayed_Payment' in data.columns:
        data['Delayed_Payments_per_Month'] = data['Num_of_Delayed_Payment'] / (data['Credit_History_Months'] + 1.0)

    # 9. Estimated Interest Load
    if 'Outstanding_Debt' in data.columns and 'Interest_Rate' in data.columns:
        data['Estimated_Annual_Interest'] = data['Outstanding_Debt'] * (data['Interest_Rate'] / 100.0)

    # Drop non-predictive identifiers
    cols_to_drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True, errors='ignore')

    return data
