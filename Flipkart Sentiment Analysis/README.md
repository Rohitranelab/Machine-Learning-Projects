# 🛍️ Flipkart Reviews Sentiment Analysis

### Uncovering what customers *really* think — one review at a time.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-VADER-yellow?logo=nltk" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/WordCloud-Visualization-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Complete-success" />
</p>

---

## 📌 Overview

E-commerce platforms like **Flipkart** generate thousands of unfiltered customer reviews every day — a goldmine of insight hiding in plain text. This project builds an end-to-end **NLP pipeline** that scrapes the noise out of raw review text and turns it into a clear, data-backed verdict on customer sentiment.

Using **Natural Language Processing** and **VADER Sentiment Analysis**, the pipeline classifies real customer reviews of laptops sold on Flipkart as **Positive, Negative, or Neutral**, and visualizes the story the ratings and words are telling.

> **Business question answered:** *Are Flipkart customers happy with their laptop purchases — and can we prove it with data instead of guesswork?*

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| 📄 Reviews Analyzed | **2,304** |
| ⭐ 5-Star Ratings | **60.0%** |
| ⭐ 4-Star Ratings | **24.0%** |
| ⭐ 3-Star Ratings | **6.1%** |
| ⭐ 2-Star Ratings | **2.0%** |
| ⭐ 1-Star Ratings | **8.0%** |
| 😀 Positive Sentiment Score | 835.67 |
| 😐 Neutral Sentiment Score | 1,363.41 |
| ☹️ Negative Sentiment Score | 104.92 |
| 🏁 **Overall Verdict** | **Neutral-leaning-Positive** — reviews are largely descriptive/neutral in tone, with positive sentiment far outweighing negative |

**Takeaway:** Despite 84% of reviews carrying a 4 or 5-star rating, the sentiment engine reveals that most review *text* is neutral/descriptive (people explaining specs and use-cases) rather than emotionally charged — with genuine negativity being rare (only ~8% of the emotional signal). This is the kind of nuance a star-rating average alone would completely miss.

---

## 🧠 What This Project Demonstrates

- **End-to-end NLP workflow** — from raw, messy text to actionable business insight
- **Custom text preprocessing pipeline**: lowercasing, URL/HTML stripping, punctuation & digit removal, stopword filtering, and stemming (Snowball Stemmer)
- **Lexicon-based sentiment scoring** using NLTK's **VADER**, purpose-built for short, informal, review-style text
- **Exploratory Data Analysis (EDA)** with rating distribution and word-frequency visualizations
- **Data storytelling** — translating numeric scores into a clear, human-readable conclusion

---

## 🔍 Pipeline Walkthrough

```
Raw Reviews (CSV)
      │
      ▼
Text Cleaning ──▶ lowercase → remove URLs/HTML/punctuation/digits
      │
      ▼
Stopword Removal + Snowball Stemming
      │
      ▼
Exploratory Analysis ──▶ Rating distribution (pie chart) + Word Cloud
      │
      ▼
VADER Sentiment Scoring ──▶ Positive / Negative / Neutral per review
      │
      ▼
Aggregation & Final Verdict
```

### 1. Data Collection
Loaded **2,304 real Flipkart product reviews** (laptops) directly from a public dataset, including product name, review text, and star rating.

### 2. Text Cleaning
A custom `clean()` function strips each review down to its semantic core:
```python
"Great performance but usually it has also that gaming laptop's 
issue of battery. It can only stand for 2-3 hrs without adapter..."
                              ⬇
"great perform usual also game laptop issu batteri stand hrs 
without adapt prefer use adaptor use softwar play game"
```

### 3. Exploratory Visualization
- 🥧 **Pie chart** of the star-rating distribution to understand baseline customer satisfaction
- ☁️ **Word cloud** to surface the most frequently mentioned terms across all reviews at a glance

### 4. Sentiment Scoring with VADER
Each review is scored across three dimensions — `positive`, `negative`, and `neutral` — using NLTK's VADER `SentimentIntensityAnalyzer`, which is optimized for social-media-style and colloquial text like product reviews.

### 5. Aggregation & Verdict
Sentiment scores are summed across the entire dataset, then compared to output one clear, defensible label — **Positive**, **Negative**, or **Neutral** — representing the overall customer mood.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| NLP | NLTK (VADER, Stopwords, Snowball Stemmer), Regex |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud
```

### Run it yourself
```bash
git clone <your-repo-url>
cd flipkart-sentiment-analysis
jupyter notebook Flipkart_Sentiment_Analysis.ipynb
```

The notebook downloads its dataset directly from a public CSV source and required NLTK corpora on first run — no manual setup required.

---

## 📁 Project Structure

```
├── Flipkart_Sentiment_Analysis.ipynb   # Main analysis notebook
└── README.md                      
```

---

## 🔮 Future Enhancements

- [ ] Build a supervised classifier (Logistic Regression / Naive Bayes) trained on star ratings for comparison against VADER's lexicon-based scores
- [ ] Break down sentiment by **product** to identify specific problem SKUs
- [ ] Deploy as an interactive **Streamlit dashboard** for live review monitoring
- [ ] Extend to aspect-based sentiment analysis (e.g., battery, display, price mentioned separately)

---

## 👤 Author

### Rohit Rane

Aspiring Machine Learning Engineer | MLOps Enthusiast

---

**⭐ If you found this project interesting, consider giving it a star!**
