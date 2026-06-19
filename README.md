# 🏏 IPL Real-Time Match Prediction using Machine Learning

<div align="center">

### Predicting IPL Chase Outcomes Ball-by-Ball using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![IPL](https://img.shields.io/badge/IPL-2008--2026-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## 📌 Overview

The **IPL Real-Time Match Prediction System** is a machine learning-powered application designed to estimate the winning probability of a team chasing in the second innings of an IPL match.

Built using **Logistic Regression**, **Scikit-Learn Pipelines**, and **Streamlit**, the project processes historical IPL ball-by-ball data to generate dynamic win probabilities based on the current match situation.

By considering factors such as runs required, balls remaining, wickets in hand, venue conditions, and run rates, the system simulates the type of probability engines commonly used in modern cricket analytics platforms.

---

## ✨ Key Highlights

* 🏏 **Ball-by-Ball IPL Dataset Analysis**
* 📊 **283,000+ Deliveries Processed**
* 🧹 Team Name Standardization & Data Cleaning
* 🎯 Logistic Regression Probability Model
* ⚡ Interactive Streamlit Dashboard
* 🧠 Scikit-Learn Pipeline Integration
* 📈 Real-Time Win Probability Estimation
* 🌍 Venue-Aware Predictions
* 💾 Serialized Model Deployment using Pickle
* 🚀 ~80.4% Prediction Accuracy

---

## 📊 Dataset Summary

| Property            | Details                   |
| ------------------- | ------------------------- |
| Raw Deliveries      | 283,678                   |
| Filtered Deliveries | 114,581                   |
| Seasons Covered     | 2008 – 2026               |
| Teams Included      | 10 Current IPL Franchises |
| Match Type          | Second Innings Chases     |
| Target Variable     | Match Won / Lost          |
| Missing Values      | Removed                   |
| Encoding Method     | One-Hot Encoding          |

---

## 🏟️ Features Used by the Model

| Feature           | Description           |
| ----------------- | --------------------- |
| Batting Team      | Chasing Team          |
| Bowling Team      | Defending Team        |
| City              | Match Venue           |
| Runs Remaining    | Runs Needed to Win    |
| Balls Remaining   | Deliveries Left       |
| Wickets Remaining | Available Wickets     |
| Current Run Rate  | Current Scoring Rate  |
| Required Run Rate | Required Scoring Rate |

---

## 🤖 Machine Learning Pipeline

```mermaid
flowchart TD

A[Load IPL Dataset] --> B[Team Standardization]

B --> C[Filter Current Teams]

C --> D[Extract Second Innings Matches]

D --> E[Feature Engineering]

E --> F[One Hot Encoding]

F --> G[Train Test Split]

G --> H[Logistic Regression]

H --> I[Model Evaluation]

I --> J[Save Pipeline]

J --> K[Streamlit Deployment]

K --> L[Live Match Inputs]

L --> M[Win Probability Prediction]
```

---

## 📈 Model Performance

| Metric           | Score               |
| ---------------- | ------------------- |
| Algorithm        | Logistic Regression |
| Test Accuracy    | **80.37%**          |
| Features Used    | **8**               |
| Training Samples | **91,664**          |
| Testing Samples  | **22,917**          |

The model demonstrates strong predictive capability for T20 chase scenarios and provides probability-based insights instead of simple binary classifications.

---

## 🖥️ Streamlit Dashboard Features

### Match Context Inputs

✔ Batting Team (Chasing)

✔ Bowling Team (Defending)

✔ Match Venue

✔ Target Score

✔ Current Score

✔ Overs Completed

✔ Wickets Lost

### Live Match Metrics

📌 Runs Required

📌 Balls Remaining

📌 Current Run Rate

📌 Required Run Rate

### Prediction Output

📈 Winning Probability %

📉 Losing Probability %

🏏 Real-Time Chase Projection

---

<div align="center">

### 🏆 Bringing Data Science to Cricket Analytics

⭐ If you found this project interesting, consider giving it a star.

</div>
