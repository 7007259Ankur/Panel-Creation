# 🎓 Panel-Creation – Multi-Label Classification of Technical Panels and Research Areas

![Model Overview](Display.png)

**Author**: Ankur Gupta  
**Institution**: Indian Institute of Information Technology, Guwahati  
**Supervisor**: Dr. Subhasish Dhal  
**Date**: April 2025  
**Degree**: B.Tech in Computer Science and Engineering  
**Project Type**: Pre-Final Year Major Project

---

## 📌 Abstract

This project presents a **multi-output classification system** to automatically assign research project proposals to relevant **technical panels** and **research areas** based on their descriptions.  
Using **TF-IDF**, **Random Forest classifiers**, and **multi-label learning**, the system reaches:
- **96.2% panel assignment accuracy**
- **Hamming loss of 0.018**
- **Micro F1-score of 0.834**

This improves efficiency, reduces manual effort, and brings consistency to academic research management.

---

## 🧠 Methodology Overview

### 🔹 Dataset
- 407 real project descriptions (title + keywords)
- Labels: one technical panel + multiple research areas

### 🔹 Preprocessing
- Text combined and cleaned
- TF-IDF vectorization with bigrams (500 features)
- Label encoding + binarization

### 🔹 Model
- `RandomForestClassifier` inside `MultiOutputClassifier`
- Balanced class weights to handle label imbalance
- Custom scoring:  
  `Score = 0.3 * Panel Accuracy + 0.7 * Micro F1 (Research Areas)`

### 🔹 Optimization
- `RandomizedSearchCV` (3-fold CV)
- Best parameters:
  - `n_estimators=50`
  - `min_samples_split=5`
  - `max_depth=None`

---

## 📊 Results

| Metric              | Value    |
|---------------------|----------|
| Panel Accuracy      | 96.2%    |
| Hamming Loss        | 0.018    |
| Micro F1-Score      | 0.834    |

### 🔁 Baseline Comparison (SVM vs Our Model)

| Model     | Panel Acc | Hamming Loss | Micro F1 |
|-----------|-----------|--------------|----------|
| SVM       | 92.0%     | 0.025        | 0.907    |
| Proposed  | **96.2%** | **0.018**    | 0.834    |

✅ Higher panel accuracy  
✅ Lower error rate (Hamming loss)  
✅ Better interpretability & faster training than deep learning models

---

## 🧪 Ethical Considerations

- Dataset anonymized (no student names or IDs)
- Used balanced weights to reduce label bias

---

## 🔮 Future Work

- Replace TF-IDF with **BERT embeddings** for deeper semantic understanding  
- Apply **SHAP/LIME** for explainability  
- Use **Bayesian optimization** for better hyperparameter tuning  
- Expand dataset for better generalization

---

## 🙌 Acknowledgements

- Guided by **Dr. Subhasish Dhal**
- Developed as part of pre-final year project at **IIIT Guwahati**


