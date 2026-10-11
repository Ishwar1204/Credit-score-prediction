// Credit Score Predictor - Client Application (Report Edition: Good / Average / Bad | ₹ INR)
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initPresets();
    initSinglePrediction();
    initBatchPrediction();
    loadModelInsights();
});

let probChart = null;
let featureChart = null;

// Tab Switching
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.add("hidden"));

            btn.classList.add("active");
            document.getElementById(target).classList.remove("hidden");

            if ((target === "tab-insights" || target === "tab-report") && !featureChart) {
                loadModelInsights();
            }
        });
    });
}

// Preset Customer Profiles in Indian Rupee (₹)
const PRESETS = {
    prime: {
        Age: 38,
        Occupation: "Doctor",
        Annual_Income: 1450000,
        Monthly_Inhand_Salary: 115000,
        Num_Bank_Accounts: 2,
        Num_Credit_Card: 3,
        Interest_Rate: 7.5,
        Num_of_Loan: 1,
        loans: ["Home Equity Loan"],
        Delay_from_due_date: 0,
        Num_of_Delayed_Payment: 0,
        Changed_Credit_Limit: 25000,
        Num_Credit_Inquiries: 1,
        Credit_Mix: "Good",
        Outstanding_Debt: 28000,
        Credit_Utilization_Ratio: 14.5,
        credit_history_years: 16,
        credit_history_months: 4,
        Payment_of_Min_Amount: "No",
        Total_EMI_per_month: 12000,
        Amount_invested_monthly: 35000,
        Payment_Behaviour: "High_spent_Small_value_payments",
        Monthly_Balance: 48000
    },
    standard: {
        Age: 31,
        Occupation: "Engineer",
        Annual_Income: 720000,
        Monthly_Inhand_Salary: 55000,
        Num_Bank_Accounts: 3,
        Num_Credit_Card: 4,
        Interest_Rate: 12.0,
        Num_of_Loan: 2,
        loans: ["Auto Loan", "Personal Loan"],
        Delay_from_due_date: 4,
        Num_of_Delayed_Payment: 3,
        Changed_Credit_Limit: 12000,
        Num_Credit_Inquiries: 2,
        Credit_Mix: "Standard",
        Outstanding_Debt: 45000,
        Credit_Utilization_Ratio: 28.5,
        credit_history_years: 8,
        credit_history_months: 6,
        Payment_of_Min_Amount: "No",
        Total_EMI_per_month: 14000,
        Amount_invested_monthly: 10000,
        Payment_Behaviour: "Low_spent_Small_value_payments",
        Monthly_Balance: 16000
    },
    subprime: {
        Age: 24,
        Occupation: "Teacher",
        Annual_Income: 300000,
        Monthly_Inhand_Salary: 23000,
        Num_Bank_Accounts: 6,
        Num_Credit_Card: 7,
        Interest_Rate: 24.0,
        Num_of_Loan: 5,
        loans: ["Payday Loan", "Personal Loan", "Credit-Builder Loan"],
        Delay_from_due_date: 22,
        Num_of_Delayed_Payment: 14,
        Changed_Credit_Limit: 5000,
        Num_Credit_Inquiries: 8,
        Credit_Mix: "Bad",
        Outstanding_Debt: 82000,
        Credit_Utilization_Ratio: 48.0,
        credit_history_years: 1,
        credit_history_months: 8,
        Payment_of_Min_Amount: "Yes",
        Total_EMI_per_month: 12000,
        Amount_invested_monthly: 1000,
        Payment_Behaviour: "Low_spent_Small_value_payments",
        Monthly_Balance: 2500
    }
};

function applyPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    for (const [key, val] of Object.entries(data)) {
        if (key === "loans") {
            const checkboxes = document.querySelectorAll("input[name='loans']");
            checkboxes.forEach(cb => cb.checked = val.includes(cb.value));
        } else {
            const el = document.getElementById(key);
            if (el) el.value = val;
        }
    }

    document.getElementById("prediction-form").dispatchEvent(new Event("submit"));
}

function initPresets() {
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.preset;
            applyPreset(key);
        });
    });
}

// Single Customer Prediction
function initSinglePrediction() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const submitBtn = document.getElementById("btn-submit-predict");
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Scoring Credit Category with AI...';
        submitBtn.disabled = true;

        const selectedLoans = Array.from(document.querySelectorAll("input[name='loans']:checked")).map(cb => cb.value);
        const histY = parseInt(document.getElementById("credit_history_years").value) || 0;
        const histM = parseInt(document.getElementById("credit_history_months").value) || 0;

        const payload = {
            Age: parseFloat(document.getElementById("Age").value) || 30,
            Occupation: document.getElementById("Occupation").value,
            Annual_Income: parseFloat(document.getElementById("Annual_Income").value) || 500000,
            Monthly_Inhand_Salary: parseFloat(document.getElementById("Monthly_Inhand_Salary").value) || 40000,
            Num_Bank_Accounts: parseInt(document.getElementById("Num_Bank_Accounts").value) || 3,
            Num_Credit_Card: parseInt(document.getElementById("Num_Credit_Card").value) || 4,
            Interest_Rate: parseFloat(document.getElementById("Interest_Rate").value) || 12,
            Num_of_Loan: selectedLoans.length,
            Type_of_Loan: selectedLoans.join(", "),
            Delay_from_due_date: parseInt(document.getElementById("Delay_from_due_date").value) || 0,
            Num_of_Delayed_Payment: parseInt(document.getElementById("Num_of_Delayed_Payment").value) || 0,
            Changed_Credit_Limit: parseFloat(document.getElementById("Changed_Credit_Limit").value) || 0,
            Num_Credit_Inquiries: parseFloat(document.getElementById("Num_Credit_Inquiries").value) || 0,
            Credit_Mix: document.getElementById("Credit_Mix").value,
            Outstanding_Debt: parseFloat(document.getElementById("Outstanding_Debt").value) || 0,
            Credit_Utilization_Ratio: parseFloat(document.getElementById("Credit_Utilization_Ratio").value) || 30,
            Credit_History_Age: `${histY} Years and ${histM} Months`,
            Payment_of_Min_Amount: document.getElementById("Payment_of_Min_Amount").value,
            Total_EMI_per_month: parseFloat(document.getElementById("Total_EMI_per_month").value) || 0,
            Amount_invested_monthly: parseFloat(document.getElementById("Amount_invested_monthly").value) || 0,
            Payment_Behaviour: document.getElementById("Payment_Behaviour").value,
            Monthly_Balance: parseFloat(document.getElementById("Monthly_Balance").value) || 0
        };

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await resp.json();
            if (data.status === "success") {
                displayPredictionResult(data.data, payload);
            } else {
                alert("Error predicting: " + (data.message || "Unknown error"));
            }
        } catch (err) {
            console.error(err);
            alert("Network error: Could not reach server.");
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });

    setTimeout(() => {
        applyPreset("standard");
    }, 400);
}

function displayPredictionResult(res, payload) {
    const scoreCard = document.getElementById("result-card");
    scoreCard.classList.remove("hidden");

    const scoreTitle = document.getElementById("result-score-title");
    const scoreBadge = document.getElementById("result-score-badge");
    const scoreDesc = document.getElementById("result-score-desc");
    const gaugePath = document.getElementById("gauge-fill-path");
    const gaugeText = document.getElementById("gauge-percentage-text");
    const cibilText = document.getElementById("cibil-score-value");

    const score = res.credit_score;
    const confidence = Math.round(res.confidence * 100);
    const cibil = res.cibil_score || 720;

    scoreTitle.textContent = score.toUpperCase();
    gaugeText.textContent = `${confidence}%`;
    if (cibilText) {
        cibilText.textContent = `${cibil} / 900`;
    }

    scoreBadge.className = "inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold tracking-wide";
    
    let strokeColor = "#3b82f6";
    if (score === "Good") {
        scoreBadge.classList.add("badge-good");
        scoreBadge.innerHTML = '<i class="fas fa-check-circle mr-2"></i> GOOD CREDIT (LOW RISK)';
        scoreDesc.textContent = "High creditworthiness profile! Eligible for optimal interest rates, high credit limits, and priority approvals.";
        strokeColor = "#10b981";
    } else if (score === "Average") {
        scoreBadge.classList.add("badge-standard");
        scoreBadge.innerHTML = '<i class="fas fa-balance-scale mr-2"></i> AVERAGE CREDIT (MODERATE RISK)';
        scoreDesc.textContent = "Acceptable repayment behavior with standard credit terms. Lowering debt and on-time dues can elevate to Good.";
        strokeColor = "#f59e0b";
    } else {
        scoreBadge.classList.add("badge-poor");
        scoreBadge.innerHTML = '<i class="fas fa-exclamation-triangle mr-2"></i> BAD CREDIT (HIGH RISK)';
        scoreDesc.textContent = "Significant risk indicators flagged. Elevated debt, frequent delayed payments, or high revolving card utilization.";
        strokeColor = "#ef4444";
    }

    // Circular Gauge Animation
    const circumference = 264;
    const offset = circumference - (confidence / 100) * circumference;
    gaugePath.style.stroke = strokeColor;
    gaugePath.style.strokeDashoffset = offset;

    // Render Health Meter Ratios (EMI, Utilization, Debt Burden)
    if (payload) {
        const sal = payload.Monthly_Inhand_Salary || 1;
        const emiPct = Math.min(100, Math.round(((payload.Total_EMI_per_month || 0) / sal) * 100));
        const utilPct = Math.min(100, Math.round(payload.Credit_Utilization_Ratio || 0));
        const debtRatio = Math.min(100, Math.round(((payload.Outstanding_Debt || 0) / (payload.Annual_Income || 1)) * 100));

        const emiEl = document.getElementById("bar-emi-burden");
        const utilEl = document.getElementById("bar-card-util");
        const debtEl = document.getElementById("bar-debt-burden");

        if (emiEl) {
            emiEl.style.width = `${emiPct}%`;
            document.getElementById("val-emi-burden").textContent = `${emiPct}%`;
            emiEl.className = emiPct > 45 ? "h-2 rounded-full bg-red-500" : (emiPct > 30 ? "h-2 rounded-full bg-amber-500" : "h-2 rounded-full bg-emerald-500");
        }
        if (utilEl) {
            utilEl.style.width = `${utilPct}%`;
            document.getElementById("val-card-util").textContent = `${utilPct}%`;
            utilEl.className = utilPct > 35 ? "h-2 rounded-full bg-red-500" : "h-2 rounded-full bg-emerald-500";
        }
        if (debtEl) {
            debtEl.style.width = `${debtRatio}%`;
            document.getElementById("val-debt-burden").textContent = `${debtRatio}%`;
            debtEl.className = debtRatio > 50 ? "h-2 rounded-full bg-red-500" : "h-2 rounded-full bg-blue-500";
        }
    }

    renderProbChart(res.probabilities);

    // Render Risk Flags & Advice
    const recContainer = document.getElementById("result-recommendations");
    recContainer.innerHTML = "";
    if (res.risk_factors && res.risk_factors.length > 0) {
        res.risk_factors.forEach(rf => {
            const item = document.createElement("div");
            item.className = "flex items-start text-xs text-red-300 bg-red-950/40 p-2.5 rounded-lg border border-red-800/30";
            item.innerHTML = `<i class="fas fa-exclamation-circle mt-0.5 mr-2 text-red-400"></i> <span>${rf}</span>`;
            recContainer.appendChild(item);
        });
    } else {
        const item = document.createElement("div");
        item.className = "flex items-start text-xs text-emerald-300 bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-800/30";
        item.innerHTML = `<i class="fas fa-check-circle mt-0.5 mr-2 text-emerald-400"></i> <span>Zero critical risk flags detected. Customer demonstrates excellent financial discipline!</span>`;
        recContainer.appendChild(item);
    }
}

function renderProbChart(probs) {
    const ctx = document.getElementById("probChart").getContext("2d");
    const labels = ["Good", "Average", "Bad"];
    const values = [
        Math.round((probs["Good"] || 0) * 100),
        Math.round((probs["Average"] || 0) * 100),
        Math.round((probs["Bad"] || 0) * 100)
    ];

    if (probChart) {
        probChart.destroy();
    }

    probChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    "rgba(16, 185, 129, 0.8)",
                    "rgba(245, 158, 11, 0.8)",
                    "rgba(239, 68, 68, 0.8)"
                ],
                borderColor: [
                    "#10b981",
                    "#f59e0b",
                    "#ef4444"
                ],
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Confidence: ${ctx.raw}%`
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.06)" },
                    ticks: { color: "#9ca3af", callback: (val) => `${val}%` }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f9fafb", font: { weight: "600", size: 11 } }
                }
            }
        }
    });
}

// Batch Prediction
function initBatchPrediction() {
    const runDefaultBtn = document.getElementById("btn-run-default-batch");
    const uploadForm = document.getElementById("batch-upload-form");
    const fileInput = document.getElementById("batch-file-input");

    if (runDefaultBtn) {
        runDefaultBtn.addEventListener("click", () => executeBatchPrediction(null));
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) {
                alert("Please select a CSV file first.");
                return;
            }
            executeBatchPrediction(fileInput.files[0]);
        });
    }
}

async function executeBatchPrediction(file) {
    const batchBtn = document.getElementById("btn-batch-submit");
    const statusDiv = document.getElementById("batch-status");
    const resultsDiv = document.getElementById("batch-results");

    if (batchBtn) batchBtn.disabled = true;
    statusDiv.classList.remove("hidden");
    resultsDiv.classList.add("hidden");

    const formData = new FormData();
    if (file) {
        formData.append("file", file);
    }

    try {
        const resp = await fetch("/api/predict-batch", {
            method: "POST",
            body: formData
        });

        const data = await resp.json();
        if (data.status === "success") {
            displayBatchResults(data);
        } else {
            alert("Batch error: " + (data.message || "Failed to process"));
        }
    } catch (err) {
        console.error(err);
        alert("Batch evaluation network failure.");
    } finally {
        if (batchBtn) batchBtn.disabled = false;
        statusDiv.classList.add("hidden");
    }
}

function displayBatchResults(data) {
    const resultsDiv = document.getElementById("batch-results");
    resultsDiv.classList.remove("hidden");

    document.getElementById("batch-total-count").textContent = data.total_records.toLocaleString();
    document.getElementById("batch-count-good").textContent = (data.summary["Good"] || 0).toLocaleString();
    document.getElementById("batch-count-standard").textContent = (data.summary["Average"] || 0).toLocaleString();
    document.getElementById("batch-count-poor").textContent = (data.summary["Bad"] || 0).toLocaleString();

    document.getElementById("download-csv-data").value = data.csv_data;

    const tbody = document.getElementById("batch-table-body");
    tbody.innerHTML = "";
    data.preview.forEach(row => {
        const tr = document.createElement("tr");
        tr.className = "border-b border-gray-800 hover:bg-gray-800/40";

        let badgeClass = "badge-standard";
        if (row.Predicted_Credit_Score === "Good") badgeClass = "badge-good";
        else if (row.Predicted_Credit_Score === "Bad") badgeClass = "badge-poor";

        tr.innerHTML = `
            <td class="px-4 py-2.5 font-mono text-xs text-gray-300">${row.ID || '-'}</td>
            <td class="px-4 py-2.5">
                <span class="inline-block px-2.5 py-0.5 rounded text-xs font-bold ${badgeClass}">
                    ${row.Predicted_Credit_Score}
                </span>
            </td>
            <td class="px-4 py-2.5 text-xs text-emerald-400 font-mono">${row.Prob_Good || 0}%</td>
            <td class="px-4 py-2.5 text-xs text-amber-400 font-mono">${row.Prob_Average || 0}%</td>
            <td class="px-4 py-2.5 text-xs text-red-400 font-mono">${row.Prob_Bad || 0}%</td>
        `;
        tbody.appendChild(tr);
    });

    resultsDiv.scrollIntoView({ behavior: "smooth" });
}

// Model Insights and Benchmarks
async function loadModelInsights() {
    try {
        const resp = await fetch("/api/model-info");
        const data = await resp.json();
        const metrics = data.metrics;

        const bestKey = "LightGBM";
        if (metrics && metrics[bestKey]) {
            const m = metrics[bestKey];
            document.getElementById("metric-accuracy").textContent = `${(m.accuracy * 100).toFixed(2)}%`;
            document.getElementById("metric-macro-f1").textContent = m.macro_f1.toFixed(4);
            document.getElementById("metric-weighted-f1").textContent = m.weighted_f1.toFixed(4);

            if (m.top_features) {
                renderFeatureChart(m.top_features);
            }
        }

        // Render comparative table
        const tbody = document.getElementById("benchmark-table-body");
        if (tbody && metrics) {
            tbody.innerHTML = "";
            const modelDisplayNames = {
                "LightGBM": { name: "LightGBM (Best Model)", crown: true, class: "text-emerald-400 font-bold" },
                "XGBoost": { name: "XGBoost Classifier", crown: false, class: "text-blue-300 font-semibold" },
                "Random_Forest": { name: "Random Forest Classifier", crown: false, class: "text-purple-300 font-semibold" },
                "Decision_Tree": { name: "Decision Tree Classifier", crown: false, class: "text-amber-300" },
                "Logistic_Regression": { name: "Logistic Regression (Baseline)", crown: false, class: "text-gray-300" }
            };

            for (const [key, meta] of Object.entries(modelDisplayNames)) {
                if (metrics[key]) {
                    const m = metrics[key];
                    const tr = document.createElement("tr");
                    tr.className = "hover:bg-gray-800/30";
                    tr.innerHTML = `
                        <td class="py-2.5 px-3 ${meta.class} flex items-center gap-1.5">
                            ${meta.crown ? '<i class="fas fa-crown text-amber-400"></i>' : ''} ${meta.name}
                        </td>
                        <td class="py-2.5 px-3 font-mono">${(m.accuracy * 100).toFixed(2)}%</td>
                        <td class="py-2.5 px-3 font-mono text-gray-300">${(m.precision_macro * 100 || 0).toFixed(2)}%</td>
                        <td class="py-2.5 px-3 font-mono text-gray-300">${(m.recall_macro * 100 || 0).toFixed(2)}%</td>
                        <td class="py-2.5 px-3 font-mono text-gray-300">${m.macro_f1.toFixed(4)}</td>
                    `;
                    tbody.appendChild(tr);
                }
            }
        }
    } catch (err) {
        console.error("Could not load model insights:", err);
    }
}

function renderFeatureChart(topFeatures) {
    const ctx = document.getElementById("featureChart").getContext("2d");
    const labels = Object.keys(topFeatures);
    const values = Object.values(topFeatures);

    if (featureChart) {
        featureChart.destroy();
    }

    featureChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: "rgba(16, 185, 129, 0.75)",
                borderColor: "#10b981",
                borderWidth: 1.5,
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Predictive Weight: ${ctx.raw}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.06)" },
                    ticks: { color: "#9ca3af" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#e5e7eb", font: { size: 10 } }
                }
            }
        }
    });
}
