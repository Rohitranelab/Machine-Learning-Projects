<div align="center">

# 💼 Salary Prediction App
### Predicting Employee Salaries from Years of Experience using Machine Learning

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

**A clean, end-to-end regression pipeline — from raw data to a deployed, interactive web app.**

</div>

---

## 🧠 Project Overview

The **Salary Prediction App** is a regression-based machine learning project that estimates an employee's expected salary based on their **years of professional experience**.

> 💡 **Why it matters:** Salary estimation is a common real-world problem for HR teams, recruiters, job platforms, and job seekers alike. A simple, interpretable model like this demonstrates how a full ML workflow — data → model → deployment — can be built and shipped as a usable product.

**Real-world applications:**
- 🧑‍💼 HR & compensation benchmarking tools
- 📊 Job portals offering salary estimates to candidates
- 🎓 Career guidance platforms for students and early professionals
- 🧮 Quick "what-if" salary calculators for negotiation prep

**Expected users:** Recruiters, HR analysts, job seekers, and anyone exploring how experience correlates with compensation.

---

## ✨ Features

- ✅ Simple, intuitive Streamlit UI
- ✅ Real-time salary prediction from a single input
- ✅ Trained model loaded via serialized `.pkl` artifact
- ✅ Lightweight, dependency-minimal application
- ✅ Reproducible model-loading pipeline

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python |
| **ML Library** | Scikit-Learn |
| **Data Handling** | Pandas, NumPy |
| **Web App / Deployment** | Streamlit |
| **Model Serialization** | Pickle |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```
salary-prediction-app/
│
├── artifact/
│   └── salary.pkl          # Serialized trained regression model
│
├── data/
│   └── (dataset files)     # Raw / processed salary dataset
│
├── notebooks/
│   └── (training notebook) # EDA, preprocessing & model training
│
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Project dependencies
└── README.md                # Project documentation
```

---

## 🔄 Workflow

```
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Preparation (Years of Experience)
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Serialization (salary.pkl)
      ↓
Streamlit Deployment
      ↓
Real-time Prediction
```

---

## 📊 Dataset

> ⚠️ Dataset contents were not available during README generation.

| Detail | Value |
|---|---|
| **Source** | [Salary_Data](https://www.kaggle.com/datasets/sivaram1987/salary-data) |
| **Target Variable** | Salary |
| **Input Feature(s)** | Years of Experience |

---

## 🤖 Models Used

| Model | Purpose |
|---|---|
|Logistic Regression|Predicting salary from years of experience|

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| R² Score | *0.9772* |
| MAE | *2891.18* |
| RMSE | *3849.25* |

---

## ⚙️ Installation

Clone the repository and set up your environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/salary-prediction-app.git

# Navigate to the project directory
cd salary-prediction-app

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

Enter **years of experience** and click **Predict Salary** to see the estimated result instantly.

---

## 🔢 Example Prediction

| Years of Experience | Predicted Salary |
|---|---|
| 2.0 | ₹ *45,300.76* |
| 5.0 | ₹ *73,546.29* |
| 10.0 | ₹ *120,622.19* |

---

## 🔧 Configuration

| Parameter | Location | Description |
|---|---|---|
| `min_value` | `app.py` | Minimum allowed years of experience (currently `0.0`) |
| `step` | `app.py` | Increment step for the input field (currently `0.5`) |
| Model path | `app.py` | Path to serialized model (`artifact/salary.pkl`) |

---

## 🚀 Future Improvements

- [ ] Add multiple input features (education, job role, location, skills)
- [ ] Deploy on Streamlit Community Cloud / Render / AWS
- [ ] Containerize with Docker
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Add model explainability (SHAP)
- [ ] Add automated testing for the prediction pipeline
- [ ] Add model monitoring & retraining pipeline
- [ ] Publish evaluation metrics & experiment tracking (MLflow)

---

## 👤 Author

**Rohit Rane**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://your-portfolio.com)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

</div>

---

## 🌟 Why This Project Stands Out

> ✔ End-to-end ML workflow — from data to deployment
> ✔ Clean, minimal, and readable codebase
> ✔ Interactive, user-friendly Streamlit interface
> ✔ Lightweight dependencies, easy to set up and run
> ✔ Clear separation of model artifact, data, and app logic
> ✔ Professional documentation ready for portfolio presentation

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>
