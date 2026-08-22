# AI-Powered Hotel Booking Cancellation Prediction and Revenue Protection System

## 📌 Project Overview

The **AI-Powered Hotel Booking Cancellation Prediction and Revenue Protection System** is a machine learning project designed to predict the probability of hotel booking cancellations and help hotel management identify potential revenue at risk.

The system analyzes historical hotel booking information, applies data preprocessing and feature engineering, trains multiple machine learning models, and selects the best-performing model for cancellation prediction.

The predicted cancellation probability is then combined with booking value to estimate potential revenue at risk and categorize bookings according to their cancellation risk.

---

## 🎯 Objectives

The main objectives of this project are:

- Predict whether a hotel booking is likely to be cancelled.
- Estimate the probability of cancellation for each booking.
- Compare multiple machine learning classification models.
- Select and tune the best-performing model.
- Explain model predictions using SHAP.
- Estimate potential revenue at risk from likely cancellations.
- Categorize bookings into different risk levels.
- Provide business-oriented actions to support hotel revenue protection.
- Provide an interactive Streamlit application for practical use.

---

## 📊 Dataset

The project uses the **Hotel Booking Demand dataset**, containing historical hotel reservation information.

The original dataset contains:

- **119,390 bookings**
- **32 columns**

The target variable is:

`is_canceled`

where:

- `0` = Booking was not cancelled
- `1` = Booking was cancelled

During preprocessing, unsuitable and leakage-prone columns were removed, including:

- `reservation_status`
- `reservation_status_date`

This prevents information about the final booking outcome from being directly available to the prediction model.

---

## 🧹 Data Preprocessing

The following preprocessing activities were performed:

- Dataset inspection
- Data type analysis
- Missing-value analysis
- Duplicate analysis
- Invalid booking investigation
- Removal of completely invalid guest records
- Removal of data leakage
- Handling of missing values
- Categorical variable processing
- Numerical feature processing
- Train-test splitting
- Feature preprocessing using machine learning pipelines

---

## 🔧 Feature Engineering

Relevant booking characteristics were prepared for machine learning.

The project considers booking-related information such as:

- Lead time
- Number of adults
- Number of children
- Number of babies
- Number of previous cancellations
- Number of previous bookings
- Booking changes
- Days in waiting list
- Required car parking spaces
- Total special requests
- Average Daily Rate (ADR)
- Deposit type
- Customer type
- Hotel type
- Market segment
- Distribution channel
- Previous booking history
- Other available reservation characteristics

Feature engineering helps the model identify patterns associated with cancellation behavior.

---

## 🤖 Machine Learning Models

Multiple classification models were evaluated, including:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM

The models were evaluated using classification and ranking metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Cross-validation

---

## 🏆 Final Model

The final selected model is **Tuned XGBoost**.

The tuned XGBoost model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 85.72% |
| Precision | 76.23% |
| Recall | 70.31% |
| F1-Score | 73.15% |
| ROC-AUC | 92.38% |
| PR-AUC | 82.67% |

The model achieved a **5-fold cross-validation mean ROC-AUC of approximately 0.9208**, indicating consistent predictive performance across the validation folds.

---

## 🔍 Explainable AI

SHAP (SHapley Additive exPlanations) was used to improve model interpretability.

SHAP helps identify:

- Which features have the greatest influence on cancellation predictions.
- Which factors increase cancellation probability.
- Which factors decrease cancellation probability.
- How individual booking characteristics contribute to a prediction.

This makes the system more transparent and useful for business decision-making.

---

## 💰 Revenue Protection System

The project goes beyond simply predicting cancellations.

For each booking, the system estimates potential financial risk using:

**Revenue at Risk = Estimated Booking Value × Cancellation Probability**

The project calculated an estimated:

- **Total booking value:** ₹69,21,357.85
- **Expected revenue at risk:** ₹23,03,555.87
- **Estimated risk percentage:** 33.28%

These values demonstrate how cancellation prediction can be connected to financial decision-making.

---

## 🚦 Risk Categorization

Bookings are categorized according to their predicted cancellation probability.

The system can identify bookings that require different levels of attention, such as:

- Low Risk
- Medium Risk
- High Risk

High-risk bookings can be prioritized for revenue-protection actions.

---

## 💡 Business Recommendations

The system can support hotel management by helping them:

- Identify high-risk bookings early.
- Prioritize bookings with significant potential revenue impact.
- Monitor cancellation-prone reservations.
- Consider appropriate confirmation or follow-up strategies.
- Focus revenue-protection efforts on high-value risky bookings.
- Use predicted cancellation probabilities for better operational planning.

The system is intended as a **decision-support tool** rather than an automatic replacement for hotel management decisions.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application called **HotelGuard**.

The application uses the trained model to provide a practical interface for cancellation-risk prediction and revenue-risk analysis.

### Main project files

```text
hotelguard_ai/
│
├── app.py
├── check_model.py
├── hotelguard_final_model.pkl
├── requirements.txt
├── README.md
└── Video Project 1.mp4