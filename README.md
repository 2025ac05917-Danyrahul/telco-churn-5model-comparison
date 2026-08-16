# Customer Churn Prediction — Multi-Model Classification

## a. Problem Statement
Telecom companies lose significant revenue to customer churn. This project builds
and compares multiple classification models to predict whether a customer will
churn (leave the service) based on their account, billing, and service usage
attributes, and exposes the models through an interactive Streamlit web app for
evaluation on held-out test data.

## b. Dataset Description
- **Source:** Telco Customer Churn dataset (public, originally released by IBM
  Sample Data Sets; commonly hosted on Kaggle as "Telco Customer Churn").
- **Instances:** 7,043 customers
- **Features:** 21 raw columns (after dropping the ID column and adding 2
  lightweight engineered features — `AvgMonthlySpend`, `IsNewCustomer` — the
  model sees 21 usable features: 6 numeric, 15 categorical), which satisfies
  the assignment's minimum of 12 features and 500 instances.
- **Target:** `Churn` — binary (`Yes` / `No`)
- **Feature groups:**
  - Demographics: gender, SeniorCitizen, Partner, Dependents
  - Account info: tenure, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
  - Services subscribed: PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies
- **Preprocessing:** missing `TotalCharges` values (new customers with 0
  tenure) imputed with the median; numeric features standardized; categorical
  features one-hot encoded via a `ColumnTransformer` fit only on the training
  split to avoid leakage.
- **Split:** 80% train / 20% test (stratified on the target), random_state=42.
  The 20% test split (1,409 rows) is exported as `test_data.csv` and is what
  the Streamlit app is designed to evaluate.

## c. GitHub Repository Link
`<PASTE YOUR GITHUB REPO URL HERE AFTER YOU PUSH>`

## d. Models Used

### Comparison Table (on the held-out test set, 1,409 rows)

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression   | 0.8048   | 0.8458 | 0.6656    | 0.5321 | 0.5914 | 0.4703 |
| Decision Tree         | 0.7821   | 0.7989 | 0.6209    | 0.4599 | 0.5284 | 0.3983 |
| kNN                   | 0.7850   | 0.8221 | 0.6041    | 0.5508 | 0.5762 | 0.4334 |
| Naive Bayes           | 0.6955   | 0.8135 | 0.4593    | 0.8289 | 0.5910 | 0.4209 |
| Random Forest (Ensemble) | 0.7963 | 0.8349 | 0.6408    | 0.5294 | 0.5798 | 0.4505 |

*(Regenerated automatically by `model/train_models.py` — see `model/metrics.csv`
for the source of truth if you re-run training.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance of accuracy, AUC, and MCC. The churn signal in this dataset (contract type, tenure, monthly charges) is largely linear/monotonic, which suits a linear decision boundary well, and it doesn't overfit on the moderate feature count after one-hot encoding. |
| Decision Tree | Weakest performer here — a single tree at this depth overfits patterns in the majority class and misses more of the minority "churn" cases (lowest recall and MCC), even with depth limited to control overfitting. |
| kNN | Competitive AUC and recall since churners tend to cluster in feature space (short tenure, month-to-month contracts), but performance is sensitive to feature scaling and the choice of k; accuracy trails the linear model slightly. |
| Naive Bayes | Highest recall by a wide margin — it's willing to flag many customers as "at risk," which is useful if the business cost of missing a churner is high, but this comes at the cost of the lowest precision and accuracy because of many false positives. Its independence assumption is also clearly violated by correlated features like tenure/TotalCharges. |
| Random Forest (Ensemble) | Second-best overall — averaging many trees reduces the variance/overfitting problem seen in the single Decision Tree and lifts AUC and MCC close to Logistic Regression, though it still slightly under-predicts the minority churn class relative to Logistic Regression's recall/precision balance. |
| **Overall Winner for this dataset** | **Logistic Regression** — highest Accuracy, AUC, and MCC, with a solid Precision/Recall balance. Given the near-linear separability of churn drivers (tenure, contract type, charges) in this dataset, a simple linear model outperforms the more complex ones, which is a useful reminder that model complexity should match the underlying data structure. |

## Repository Structure
```
project-folder/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test split used for evaluation
│-- telco.csv                  # full raw dataset (for reference/reproducibility)
│-- model/
│   │-- train_models.py        # trains all 5 models, saves artifacts + metrics
│   │-- metrics.csv            # generated comparison table
│   │-- *.joblib                # trained models + fitted preprocessor
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py     # regenerates models/metrics (already included)
streamlit run app.py
```
Then in the sidebar, upload `test_data.csv` and pick a model from the dropdown.

## Deployment
Deployed on Streamlit Community Cloud: `<PASTE YOUR LIVE APP LINK HERE>`

## BITS Virtual Lab Execution
Screenshot of the app running on BITS Virtual Lab: `<INSERT SCREENSHOT / LINK HERE>`
