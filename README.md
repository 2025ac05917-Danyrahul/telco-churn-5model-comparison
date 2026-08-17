# Customer Churn Prediction — Multi-Model Classification

## a. Problem Statement
Telecom companies lose a lot of customers every year, and this is called "churn."
In this project, I built machine learning models to predict whether a customer
will leave the telecom company (churn) or stay, based on their account details,
billing information, and the services they use. I also built a simple web app
using Streamlit so anyone can upload test data and see how the models perform.

## b. Dataset Description
- **Source:** Telco Customer Churn dataset. This is a public dataset originally
  shared by IBM and is also commonly found on Kaggle under the name "Telco
  Customer Churn."
- **Number of rows:** 7,043 customers
- **Number of features:** 21 columns are used to train the models (after
  removing the customer ID column, which is not useful for prediction, and
  adding 2 new columns I created: `AvgMonthlySpend` and `IsNewCustomer`). Out
  of these, 6 are numeric and 15 are categorical.
- **Target column:** `Churn` — this tells us if the customer left (`Yes`) or
  stayed (`No`).
- **Main groups of features:**
  - Customer details: gender, SeniorCitizen, Partner, Dependents
  - Account details: tenure, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
  - Services used: PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies
- **Data cleaning done:**
  - Some `TotalCharges` values were missing for new customers, so I filled
    them in using the median value.
  - Numeric columns were scaled so all values are on a similar range.
  - Categorical columns were converted into numbers using one-hot encoding.
- **Train/Test split:** I split the data into 80% for training and 20% for
  testing, using `random_state=42` so the split stays the same every time.
  This test part (1,409 rows) was saved as `test_data.csv` and is the file
  used inside the Streamlit app.

## c. GitHub Repository Link
https://github.com/danyrahul/telco-churn-5model-comparison

## d. Models Used

### Comparison Table (results on the 1,409-row test set)

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression   | 0.8048   | 0.8458 | 0.6656    | 0.5321 | 0.5914 | 0.4703 |
| Decision Tree         | 0.7821   | 0.7989 | 0.6209    | 0.4599 | 0.5284 | 0.3983 |
| kNN                   | 0.7857   | 0.8220 | 0.6053    | 0.5535 | 0.5782 | 0.4357 |
| Naive Bayes           | 0.6955   | 0.8135 | 0.4593    | 0.8289 | 0.5910 | 0.4209 |
| Random Forest (Ensemble) | 0.7963 | 0.8349 | 0.6408    | 0.5294 | 0.5798 | 0.4505 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model gave the best overall results — highest accuracy, AUC, and MCC. Since churn in this data mostly depends on things like contract type and tenure in a fairly straightforward way, a simple linear model was able to capture the pattern well without overfitting. |
| Decision Tree | This was the weakest model. A single tree tends to overfit and struggled the most to correctly find customers who actually churned (lowest recall and MCC), even though I limited how deep the tree could grow. |
| kNN | Performed decently, with a good AUC and recall — this makes sense because customers who churn often have similar patterns (like short tenure or month-to-month contracts) and end up close to each other. But its accuracy was a bit lower than Logistic Regression. |
| Naive Bayes | This model found the most actual churners (highest recall), which is useful if missing a churner is costly for the business. But it also gave many false alarms, so its precision and overall accuracy were the lowest. Naive Bayes assumes features are independent, which is not really true here (e.g., tenure and TotalCharges are related), and that hurts its performance. |
| Random Forest (Ensemble) | This was the second-best model. By combining many decision trees, it fixed most of the overfitting problem seen in the single Decision Tree and got results close to Logistic Regression. |
| **Overall Winner for this dataset** | **Logistic Regression** — it had the best accuracy, AUC, and MCC, with a good balance of precision and recall. This shows that for this dataset, the churn patterns are simple enough that a basic linear model works better than more complex ones. |
