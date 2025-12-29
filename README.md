# News Categorization using Title-Based Text Analysis

[![Project Status](https://img.shields.io/badge/Status-Complete-success.svg)](https://github.com/Samyan1Sharma/News-Categorization)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Samyan1Sharma/News-Categorization/blob/main/LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Samyan1Sharma/News-Categorization/blob/main/CONTRIBUTING.md)
![Machine Learning](https://img.shields.io/badge/Type-Machine_Learning-blue)
![NLP](https://img.shields.io/badge/Focus-Natural_Language_Processing-brightgreen)
![Python](https://img.shields.io/badge/Language-Python_3.10-blue)

[cite_start]A high-performance Machine Learning pipeline designed to categorize news articles using **only titles**[cite: 5, 599]. [cite_start]By bypassing full-text analysis, this approach reduces computational overhead by 70–80% [cite: 481] [cite_start]while achieving a peak accuracy of **97.86%**[cite: 366, 513].

## 📌 Overview
[cite_start]Digital newsrooms require rapid categorization to improve information lookup and user experience[cite: 4]. [cite_start]While conventional systems rely on full-text analysis—which is time-consuming and resource-heavy [cite: 15][cite_start]—this project utilizes headings to provide the "purest and most structured information"[cite: 16]. [cite_start]This research proves that minimal textual input, when processed effectively, can yield near state-of-the-art results[cite: 21, 516].

## 🎯 Objectives
* [cite_start]**Automated Categorization**: Classify news into five topics using only minimal title-based input[cite: 63, 509].
* [cite_start]**Computational Efficiency**: Cut processing needs by 70-80% and feature dimensionality by 60% compared to full-text methods[cite: 481, 482].
* [cite_start]**Real-time Viability**: Achieve low-latency inference (0.0014s per title) suitable for streaming APIs and real-time news filtering[cite: 369, 515].
* [cite_start]**Scalable Deployment**: Create a lightweight solution that trains significantly faster (0.036s) than deep learning alternatives[cite: 470, 495].

## 📊 Performance Highlights
[cite_start]The **SGDClassifier with TF-IDF** features emerged as the optimal configuration for accuracy and speed[cite: 258, 386].

| Model | Vectorizer | Accuracy | F1-Score | Training Time | Prediction Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SGDClassifier** | **TF-IDF** | **97.86%** | **0.9785** | **0.0363s** | **0.0014s** |
| **LinearSVC** | **TF-IDF** | 97.59% | 0.9758 | 0.1153s | 0.0008s |
| **Logistic Regression** | **TF-IDF** | 97.59% | 0.9759 | 1.5440s | 0.0014s |
| **Multinomial NB** | **BoW** | 97.32% | 0.9731 | 0.0173s | 0.0037s |
| **Random Forest** | **TF-IDF** | 96.51% | 0.9650 | 2.5368s | 0.0797s |

> [cite_start]*Source: Comparative performance benchmarks on the BBC News dataset[cite: 305, 487].*

## ✨ Key Features
* [cite_start]**Title-Only Analysis**: High accuracy achieved without requiring the expensive processing of article bodies[cite: 6, 370].
* [cite_start]**High-Speed Inference**: System supports processing approximately 700 titles per second on standard hardware[cite: 376].
* [cite_start]**Feature Optimization**: Proven superiority of **TF-IDF** over **Bag-of-Words** for short-text classification[cite: 247, 511].
* [cite_start]**Robust Preprocessing**: Comprehensive NLP pipeline including tokenization, stopword removal, and stemming[cite: 8, 211].

## 🛠 Technical Implementation

### Dataset
* [cite_start]**Source**: BBC News Train dataset[cite: 173].
* [cite_start]**Size**: 1,490 news articles[cite: 198].
* [cite_start]**Categories**: Business, Politics, Sport, Entertainment, and Technology [cite: 180-184].



### Preprocessing Workflow
[cite_start]The system standardizes raw titles through the following steps [cite: 70-74]:
1.  [cite_start]**Lowercasing**: Ensures uniformity across the dataset[cite: 85].
2.  [cite_start]**Punctuation Removal**: Removes noise from textual data[cite: 88].
3.  [cite_start]**Tokenization**: Splits titles into separate units/words[cite: 82].
4.  [cite_start]**Stopword Removal**: Drops common words that don't contribute to category meaning[cite: 78].
5.  [cite_start]**Text Stemming**: Reduces words to their base roots for standardized analysis[cite: 92].

### Model Training & Evaluation
* [cite_start]**Feature Extraction**: Comparison between TF-IDF (Term Frequency-Inverse Document Frequency) and Bag-of-Words[cite: 252].
* [cite_start]**Algorithms**: Evaluated 5 supervised machine learning models using Scikit-learn[cite: 243, 251].
* [cite_start]**Validation**: Used a 75/25 train-test split with confusion matrix analysis to identify semantic overlaps[cite: 199, 241, 388].



## 🧪 Model Inference Example
```python
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('best_model.sav', 'rb'))

# Example title for classification
sample_headline = "Microsoft released new Laptop."

# The prediction routine handles preprocessing automatically
prediction = predict_news_category_best(sample_headline)
print(f"Predicted Category: {prediction}") 
# Output: Tech
