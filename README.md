# 💳 CrediPulse Bharat AI - Credit Score Prediction Engine (₹ INR)

An enterprise-grade Machine Learning credit rating system and web application tailored for Indian Rupee (**₹ INR**) financial profiles. Built with a high-accuracy **Tuned LightGBM** model achieving **78.91% Accuracy** across 46 engineered financial features.

---

## 📌 Performance Benchmarks (Enhanced Model)

| Model | Accuracy | Macro F1-Score | Weighted F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM Tuned** | **78.91%** | **0.7762** | **0.7890** | **Best Selected (Active)** |
| 🥈 **XGBoost Tuned** | 77.05% | 0.7549 | 0.7705 | Runner-up |
| 🥉 **CatBoost Classifier** | 73.68% | 0.7158 | 0.7367 | Baseline |

### 🔑 Top Predictive Risk Drivers
1. **`Credit_History_Months`**: Overall longevity and depth of credit history.
2. **`Changed_Credit_Limit`**: Frequency and magnitude of credit limit alterations.
3. **`Age`**: Financial maturity and earning cycle stability.
4. **`Estimated_Annual_Interest`**: Annual interest load relative to outstanding obligations.
5. **`Delay_Severity_Index`**: Compound score of delay days multiplied by delayed payment count.
6. **`Delayed_Payments_per_Month`**: Rate of delinquency over total history months.
7. **`Outstanding_Debt`**: High debt burden in ₹ INR.
8. **`Total_EMI_per_month`**: Total monthly installment obligation in ₹ INR.
9. **`EMI_to_Salary_Ratio`**: Percentage of in-hand salary consumed by EMIs (RBI threshold < 40%).
10. **`Credit_Utilization_Ratio`**: Revolving credit utilization percentage (CIBIL recommended < 30%).

---

## 💻 How to Open and Run in VS Code (Visual Studio Code)

All project files are already located in your workspace: `c:\Users\Ishwar\Downloads\credit`.
VS Code configuration files (`.vscode/launch.json`, `.vscode/settings.json`, and `.vscode/tasks.json`) are already pre-configured for you!

### Step 1: Open Folder in VS Code
You can open the project in VS Code using any of these methods:
- **Method A (Quick)**: Double-click [`open_in_vscode.bat`](file:///c:/Users/Ishwar/Downloads/credit/open_in_vscode.bat) in the project folder.
- **Method B (VS Code Menu)**: Open Visual Studio Code $\rightarrow$ click **File** $\rightarrow$ **Open Folder...** $\rightarrow$ select `c:\Users\Ishwar\Downloads\credit`.
- **Method C (Command Prompt / PowerShell)**:
  ```bash
  cd c:\Users\Ishwar\Downloads\credit
  code .
  ```

### Step 2: Run with 1-Click Debugger (F5)
1. In VS Code, press **`Ctrl + Shift + D`** (or click the **Run and Debug** icon on the left sidebar).
2. At the top of the Run & Debug panel, select any of the ready-to-run configurations from the dropdown:
   - **`🌐 Run Web App (http://localhost:5000)`** $\rightarrow$ Starts the web application on port 5000.
   - **`💳 Run Interactive CLI (cli.py)`** $\rightarrow$ Launches the interactive console assessment with ₹ INR inputs.
   - **`⚡ Run Batch Prediction (test.csv)`** $\rightarrow$ Generates `test_predictions.csv` for `test.csv`.
   - **`🧠 Retrain Model (src/train.py)`** $\rightarrow$ Runs model training and updates benchmarks.
3. Press **`F5`** or click the green **Play** button!

### Step 3: Run via VS Code Integrated Terminal
Press **``Ctrl + ` ``** to open the integrated terminal in VS Code and run:
- **Start Web Dashboard**:
  ```bash
  python web_app.py
  ```
  Then open `http://localhost:5000` in your browser.
- **Run Terminal Interactive CLI**:
  ```bash
  python cli.py
  ```
- **Batch Predict on CSV**:
  ```bash
  python -m src.predict --file test.csv --output test_predictions.csv
  ```

---

## 🌐 Web Application Features (CrediPulse Bharat AI)

- **Indian Rupee (₹ INR) Context**: All figures (Annual Income, In-hand Salary, Outstanding Debt, EMIs, Balance) formatted in ₹.
- **Estimated CIBIL Scale**: Provides an estimated credit score on the 300–900 scale alongside the classification (Good, Standard, Poor).
- **Financial Health Ratios**: Dynamic progress meters for:
  - EMI Burden (% of monthly salary)
  - Credit Card Utilization (% of total limit)
  - Debt-to-Annual-Income Ratio
- **Interactive Preset Borrowers**: Test 🌟 Prime Borrower, ⚖️ Average Borrower, and 🚨 High Risk Borrower with a single click.
- **Batch File Evaluation**: Upload custom CSVs or evaluate built-in `test.csv` with a 1-click download of scored results.
- **Model Analytics Tab**: Live chart of top 15 feature importances and comparative performance table.

---

## 📂 Project Structure

```text
c:\Users\Ishwar\Downloads\credit\
├── .vscode/
│   ├── launch.json          <- 1-click F5 Run configurations
│   ├── settings.json        <- Python interpreter path and terminal settings
│   └── tasks.json           <- VS Code build and run tasks
├── src/
│   ├── preprocessing.py     <- 46 financial domain features and dirty data cleaning
│   ├── train.py             <- Model training (Tuned LightGBM, CatBoost, Tuned XGBoost)
│   └── predict.py           <- Single & batch inference engine (₹ INR & CIBIL score)
├── templates/
│   └── index.html           <- Responsive FinTech dark-mode web dashboard
├── static/
│   ├── css/style.css        <- FinTech glassmorphism styling
│   └── js/main.js           <- Client controller, presets, and Chart.js integration
├── models/
│   ├── best_credit_score_model.joblib  <- Trained 78.91% LightGBM model bundle
│   └── training_metrics.json          <- Accuracy, F1 scores & feature importances
├── web_app.py               <- Flask web server backend
├── cli.py                   <- Terminal interactive CLI (₹ INR)
├── open_in_vscode.bat       <- Batch script to open project in VS Code
├── run_website.bat          <- Batch script to launch website on Windows
├── train.csv                <- 100,000 training records
└── test.csv                 <- 50,000 evaluation records
```
