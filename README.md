# Credit-Card-Default-Risk-Analysis
Machine learning project for analyzing and predicting credit card default risk using Logistic Regression and XGBoost.

Predicting whether a credit card customer will **default on their payment next month**, using the UCI *Default of Credit Card Clients* dataset (Taiwan, 30,000 customers). The project covers data cleaning, exploratory analysis, feature engineering, model training and tuning (Logistic Regression vs. XGBoost), evaluation, and an interactive risk-prediction dashboard built with `ipywidgets` and Plotly.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Interactive Dashboard](#-interactive-dashboard)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Known Limitations & Future Work](#-known-limitations--future-work)

---

## Problem Statement

Credit card issuers need to identify customers likely to default so they can manage risk, set credit limits, and intervene early. This project builds and compares classification models that predict `DEFAULT_NEXT_MONTH` (1 = default, 0 = no default) from a customer's demographics, credit limit, six months of repayment history, bill amounts, and payment amounts.

## Dataset

- **Source:** [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Size:** 30,000 records, 24 original columns
- **Target:** `DEFAULT_NEXT_MONTH` (~22.1% default rate → imbalanced classes)

| Group | Columns |
|-------|---------|
| Credit | `LIMIT_BAL` (credit limit, NTD) |
| Demographics | `GENDER`, `EDUCATION`, `MARITAL_STATUS`, `AGE` |
| Repayment status | `PAY_1` … `PAY_6` (Sep → Apr; -2 = no consumption, -1 = paid duly, 0 = revolving credit, 1–9 = months of delay) |
| Bill amounts | `BILL_AMT1` … `BILL_AMT6` |
| Payment amounts | `PAY_AMT1` … `PAY_AMT6` |

> The dataset is not included in this repo. Download it from the link above and update the file path in the notebook.

## Project Workflow

### 1. Data Cleaning & Pre-processing
- Standardised column names (`SEX → GENDER`, `MARRIAGE → MARITAL_STATUS`, `PAY_0 → PAY_1`, etc.) and dropped the `ID` column
- Mapped numeric codes to readable categories (gender, education, marital status)
- Merged undocumented category codes into `OTHER` (education `0/5/6`, marital status `0`)
- One-hot encoded categorical variables (`drop_first=True`)

### 2. Exploratory Data Analysis
- Class distribution (default vs. non-default)
- Default rate by gender, education, and marital status
- Credit limit distribution and age distribution by default status
- Correlation heatmap
- Payment-delay distribution across the six months
- Bill amount vs. payment amount scatter plot

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `num_delayed_months`, `max_delay`, `avg_delay` | Summary of repayment delays across `PAY_1`–`PAY_6` |
| `pay_ratio_1` … `pay_ratio_6` | Monthly payment ÷ bill amount |
| `bill_trend`, `pay_trend` | Change in bill / payment between month 6 and month 1 |
| `avg_bill_amt`, `credit_utilization` | Average bill and average bill ÷ credit limit |
| `total_payment`, `total_bill`, `payment_to_bill_ratio` | Six-month totals and overall repayment ratio |

### 4. Model Training
- Stratified 80/20 train–test split (`random_state=42`) — 24,000 train / 6,000 test rows, default rate preserved at 22.12% in both
- Numerical features standardised with `StandardScaler` (infinite values handled, then median-imputed)
- Models: **Logistic Regression** and **XGBoost**

### 5. Hyperparameter Tuning
- **Logistic Regression:** `GridSearchCV` over penalty (`l1`, `l2`, `elasticnet`), `C`, and `l1_ratio` (5-fold CV, ROC-AUC)
- **XGBoost:** `RandomizedSearchCV` over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma` (20 iterations, 5-fold CV, ROC-AUC)

### 6. Evaluation
Classification reports, confusion matrices, ROC curves, precision–recall curves, and XGBoost top-10 feature importances. Tuned models are saved with `joblib` (`optimized_lr_model.pkl`, `optimized_xgb_model.pkl`).

## Results

Baseline (untuned) performance on the held-out test set:

| Metric | Logistic Regression | XGBoost |
|--------|:------------------:|:-------:|
| Accuracy | 0.809 | **0.819** |
| Precision | **0.693** | 0.668 |
| Recall | 0.245 | **0.359** |
| F1-Score | 0.362 | **0.467** |
| ROC-AUC | 0.710 | **0.779** |

**Takeaways**
- XGBoost outperforms Logistic Regression on ROC-AUC, F1, and recall, so it is the stronger baseline.
- Both models have low recall on the default class — a consequence of class imbalance. Since missing a defaulter is usually costlier than flagging a good customer, threshold tuning or class weighting is worth exploring.
- Tuned-model scores are printed in the notebook (`Optimized LR ROC-AUC` / `Optimized XGB ROC-AUC`); re-run the notebook to reproduce them.

## Interactive Dashboard

The final notebook cells launch an in-notebook **Credit Card Default Risk Prediction** dashboard (dark theme) where you can:

- Choose a model (Logistic Regression or XGBoost)
- Adjust credit limit, age, gender, education, marital status, six months of repayment status, bill amounts, and payment amounts using sliders and dropdowns
- View the predicted default probability and risk level, along with Plotly visualisations

The dashboard loads the saved `.pkl` models. If they are not found, it falls back to dummy models so the UI can still be demonstrated — run the training cells first to get real predictions.

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Modeling:** scikit-learn, XGBoost
- **Dashboard:** ipywidgets, IPython
- **Persistence:** joblib

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly ipywidgets joblib jupyter
```

**3. Add the dataset**
Download the UCI dataset and load it into a DataFrame named `df` at the top of the notebook, for example:
```python
df = pd.read_excel("default of credit card clients.xls", header=1)
```

**4. Run the notebook**
```bash
jupyter notebook Credit_card_default_risk_analysis.ipynb
```
Run all cells in order. The last cells launch the dashboard.

## Repository Structure

```
├── Credit_card_default_risk_analysis.ipynb   # Full analysis, modeling & dashboard
└── README.md
```

## Known Limitations & Future Work

- **Engineered features aren't used in the final models.** The one-hot encoded DataFrame (`df_encoded`) is created before the feature-engineering step, so the models train on the 26 original/encoded features. Re-running the encoding after feature engineering would let the models use the new features.
- **Class imbalance:** try `class_weight` / `scale_pos_weight`, SMOTE, or decision-threshold tuning to improve recall on defaulters.
- **Scaling:** tree models like XGBoost don't need scaled inputs; scaling is only needed for Logistic Regression.
- **Explainability:** add SHAP values for per-customer explanations.
- **Deployment:** wrap the model in a Streamlit or FastAPI app instead of an in-notebook dashboard.
- **Fairness:** the dataset includes gender, age, and marital status; any real-world use should be audited for bias.

## License

This project is for educational purposes. Dataset © UCI Machine Learning Repository (CC BY 4.0).
