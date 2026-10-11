"""
Model Training & Benchmarking Pipeline based on Ishwar Kumavat's ML Report:
'Credit Score Prediction Using Machine Learning Algorithms'
(Dept of Computer Engineering, BE Sem VII).

Implements the 4 required models + LightGBM:
1. Logistic Regression (Baseline Model)
2. Decision Tree Classifier (Nonlinear Tree Representation)
3. Random Forest Classifier (Ensemble Bagging)
4. XGBoost Classifier (Extreme Gradient Boosting)
5. LightGBM Classifier (Advanced Gradient Boosting)

Target Classes mapped according to the report:
- Good     -> 'Good'
- Standard -> 'Average'
- Poor     -> 'Bad'
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_dataframe

# Report Mappings: Poor -> Bad, Standard -> Average, Good -> Good
RAW_TO_REPORT_MAP = {'Poor': 'Bad', 'Standard': 'Average', 'Good': 'Good'}
TARGET_MAP = {'Bad': 0, 'Average': 1, 'Good': 2}
TARGET_INV_MAP = {0: 'Bad', 1: 'Average', 2: 'Good'}
CLASS_NAMES = ['Bad', 'Average', 'Good']


def train_models(data_path="train.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    print(f"--> [Module 1] Loading raw credit dataset from {data_path}...")
    df_raw = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df_raw):,} customer records.")

    print("--> [Module 2] Applying data cleaning and feature engineering...")
    df_clean = clean_dataframe(df_raw, is_train=True)

    target_col = 'Credit_Score'
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y_raw = df_clean[target_col].map(RAW_TO_REPORT_MAP)
    X = df_clean.drop(columns=[target_col])

    # Filter valid rows
    valid_mask = y_raw.isin(TARGET_MAP.keys())
    X = X[valid_mask].copy()
    y = y_raw[valid_mask].map(TARGET_MAP).astype(int)

    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = X.columns.tolist()
    print(f"Total engineered features: {len(feature_names)}")

    # Stratified Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]:,} samples | Holdout Test split: {X_val.shape[0]:,} samples")

    # Common Imputer & Scaler for Logistic Regression and Decision Tree
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    print("\n--> [Module 3] Training Machine Learning Classifiers from ML Report...")

    # Define the 4 models specified in Ishwar's report + LightGBM
    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "use_scaled": True
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=20, random_state=42),
            "use_scaled": False,
            "use_imputed": True
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "use_imputed": True
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=7,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            "use_scaled": False,
            "use_imputed": False
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=127,
                max_depth=9,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=25,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            "use_scaled": False,
            "use_imputed": False
        }
    }

    results = {}
    fitted_models = {}

    print("\n--> [Module 4] Performance Evaluation & Comparative Metrics:")
    for name, config in models.items():
        clf = config["model"]
        print(f"  Training {name}...")

        if config.get("use_scaled"):
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_val_scaled)
        elif config.get("use_imputed"):
            clf.fit(X_train_imp, y_train)
            preds = clf.predict(X_val_imp)
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)

        preds = np.asarray(preds).ravel().astype(int)

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average='macro')
        weighted_f1 = f1_score(y_val, preds, average='weighted')
        prec_macro = precision_score(y_val, preds, average='macro', zero_division=0)
        rec_macro = recall_score(y_val, preds, average='macro', zero_division=0)
        cm = confusion_matrix(y_val, preds).tolist()
        report = classification_report(
            y_val, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "confusion_matrix": cm,
            "classification_report": report
        }
        fitted_models[name] = clf

        print(f"    [{name:20s}] Acc: {acc*100:.2f}% | Prec: {prec_macro*100:.2f}% | Rec: {rec_macro*100:.2f}% | Macro F1: {macro_f1:.4f}")

    # Best model selection
    best_name = max(results, key=lambda k: (results[k]["accuracy"], results[k]["macro_f1"]))
    print(f"\n--> Selected Best Model for Deployment: {best_name} ({results[best_name]['accuracy']*100:.2f}% Accuracy)")

    best_clf = fitted_models[best_name]

    # Feature Importance for best model
    importances = {}
    if hasattr(best_clf, "feature_importances_"):
        fi = best_clf.feature_importances_
        for f, score in zip(feature_names, fi):
            importances[f] = float(score)
        sorted_fi = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15])
        results[best_name]["top_features"] = sorted_fi

    # Save artifact bundle
    artifacts = {
        "model_name": best_name,
        "model": best_clf,
        "imputer": imputer if best_name in ["Logistic_Regression", "Decision_Tree", "Random_Forest"] else None,
        "scaler": scaler if best_name == "Logistic_Regression" else None,
        "encoders": encoders,
        "feature_names": feature_names,
        "target_map": TARGET_MAP,
        "target_inv_map": TARGET_INV_MAP,
        "class_names": CLASS_NAMES,
        "evaluation_metrics": results[best_name],
        "all_model_benchmarks": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "weighted_f1": v["weighted_f1"]} for k, v in results.items()}
    }

    model_path = os.path.join(model_dir, "best_credit_score_model.joblib")
    joblib.dump(artifacts, model_path)
    print(f"--> Saved best model artifact to: {model_path}")

    # Save metrics JSON
    report_path = os.path.join(model_dir, "training_metrics.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"--> Saved comprehensive training metrics to: {report_path}")

    return results, best_name


if __name__ == "__main__":
    train_models()
