# Customer Churn Prediction

### Project Overview:
This project aims to predict whether a bank customer will churn (leave the bank) or not using machine learning techniques. The dataset contains customer demographic and account-related information, and the goal is to classify whether a customer will churn (1) or stay (0).

The project also addresses the problem of **imbalanced data** using techniques like SMOTE and advanced models such as Gradient Boosting and XGBoost.

---

### 1. **Data Understanding:**

The dataset contains ~9,800+ records with multiple features such as:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (Target Variable)

Steps for data understanding include:
- Loading dataset and inspecting first few rows
- Checking data types and missing values
- Performing basic statistical analysis
- Understanding class imbalance (Churn vs Non-Churn)

---

### 2. **Data Preprocessing:**

- **Handling Missing Values:** Checked for null values and handled appropriately  
- **Feature Engineering:** Created new feature `AgeCategory` for better segmentation  
- **Encoding:** Applied encoding techniques for categorical variables (Gender, Geography)  
- **Feature Scaling:** Applied scaling where required  
- **Handling Imbalance:** Used **SMOTE** to balance the dataset (important step)  
- **Train-Test Split:** Data split into training and testing sets  

---

### 3. **Model Building:**

Multiple classification models were trained and evaluated:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost (Best Performing Model)**

Two approaches were used:
- With SMOTE (for imbalance handling)
- Without SMOTE (for advanced boosting models)

---

### 4. **Model Evaluation:**

The models were evaluated using:

- **Accuracy**
- **F1 Score (Churn Class)**
- **Weighted F1 Score**
- **ROC-AUC Score**
- **Cross-Validation (ROC-AUC)**

Key Observations:
- Traditional models improved with SMOTE  
- Boosting models performed well without SMOTE  
- **XGBoost achieved the best performance overall**

---

### 5. **Deployment:**

The final model was deployed using **Streamlit**:

- User inputs customer details
- Model predicts churn probability
- Displays:
  - Churn / Not Churn
  - Probability score

---

### 6. **Conclusion:**

The project successfully predicts customer churn with strong performance.

- SMOTE helped improve performance of basic models  
- Boosting models (Gradient Boosting, XGBoost) performed best  
- XGBoost provided the most reliable results  

---

### Dataset:
Dataset not included due to size.  
You can use the **Bank Customer Churn dataset (Kaggle)**.

---
