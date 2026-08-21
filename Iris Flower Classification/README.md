# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using a Support Vector Machine (SVM) classifier trained on the classic Iris dataset.

---

## 📌 Project Overview

The Iris dataset is one of the most well-known datasets in machine learning. This project demonstrates a complete ML pipeline: loading data, exploratory visualization, model training, evaluation, and making predictions on new input.

---

## 📁 Project Structure

```
Iris_Flower_Classification.ipynb   # Main Jupyter notebook
README.md                          # Project documentation
```

---

## 🗂️ Dataset

- **Source:** `sklearn.datasets.load_iris` (built-in scikit-learn dataset)
- **Samples:** 150 (50 per class)
- **Features:** 4 numerical features

| Feature       | Description                        |
|---------------|------------------------------------|
| Sepal length  | Length of the sepal (cm)           |
| Sepal width   | Width of the sepal (cm)            |
| Petal length  | Length of the petal (cm)           |
| Petal width   | Width of the petal (cm)            |

- **Target Classes:**
  - `0` → Setosa
  - `1` → Versicolor
  - `2` → Virginica

---

## 🔧 Tech Stack

| Library        | Purpose                              |
|----------------|--------------------------------------|
| `pandas`       | Data manipulation                    |
| `numpy`        | Numerical operations                 |
| `matplotlib`   | Data visualization                   |
| `seaborn`      | Statistical plotting                 |
| `scikit-learn` | ML model, splitting, and evaluation  |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed. Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

```bash
jupyter notebook Iris_Flower_Classification.ipynb
```

---

## 🧪 Workflow

1. **Load Data** — Import the Iris dataset from scikit-learn
2. **Explore Data** — Visualize feature relationships using a scatter plot (Sepal length vs. Sepal width, colored by species)
3. **Split Data** — Divide into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`
4. **Train Model** — Fit a Support Vector Classifier (`SVC`) on the training data
5. **Evaluate** — Assess performance using accuracy score and a full classification report
6. **Predict** — Run inference on new, unseen flower measurements

---

## 📊 Model

- **Algorithm:** Support Vector Machine (SVC)
- **Library:** `sklearn.svm.SVC` with default hyperparameters
- **Train/Test Split:** 80% / 20%

---

## 📈 Evaluation Metrics

The model is evaluated using:

- **Accuracy Score** — Overall percentage of correct predictions
- **Classification Report** — Per-class precision, recall, and F1-score

---

## 🔮 Sample Prediction

```python
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_data)
# Output: Setosa
```

---

## 📝 Notes

- The scatter plot in the notebook uses randomly assigned labels purely for visualization color variety; the actual model trains on the true dataset labels.
- The SVM model with default settings performs very well on this linearly separable dataset, typically achieving ~97–100% accuracy.

---

## 📄 License

This project is open-source and available for educational use.
