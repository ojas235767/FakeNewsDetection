# 📰 Fake News Detection using Machine Learning

This project focuses on building a machine learning model that detects fake news articles using natural language processing (NLP) techniques. It combines TF-IDF vectorization with Logistic Regression to classify whether a given news article is real or fake.

## 🎯 Objective
To develop an accurate and accessible fake news classification system using Python and machine learning — with a user-friendly interface for real-time predictions.

## 📂 Dataset
- Source: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Records: 44,000+ news articles  
- Features include:  
  - `title`, `text`, `subject`, `date`  
  - Labeled as `REAL` or `FAKE`

## 🛠 Tools & Technologies
- Language: Python  
- Libraries:
  - `Pandas`, `NumPy` — Data manipulation  
  - `Scikit-learn` — Model building and evaluation  
  - `NLTK`, `re` — Text preprocessing  

## 🧹 Data Preprocessing
- Removal of punctuation, stopwords, and special characters  
- Lowercasing and tokenization  
- TF-IDF vectorization for converting text into numerical features

## 🤖 Model Building
- Algorithm Used: Logistic Regression  
- Why? Fast, interpretable, and effective for binary classification problems  

## ✅ Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

> Achieved high accuracy after cross-validation and hyperparameter tuning.

## 🖥️ User Interface
A lightweight interface was developed for real-time testing:
- Paste news article text or title
- Get prediction: ✅ Real or ❌ Fake

