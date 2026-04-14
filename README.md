# 🏏 IPL Match Intelligence Engine

<p align="center">
  <img src="demo_.gif" width="600">
</p>

A machine learning project that predicts the **win probability of IPL teams during a live chase** using real match data and ball-by-ball insights.

---

## 📘 Project Overview

This project predicts the **probability of a team winning an IPL match** based on real-time match conditions such as:

- Runs left  
- Balls remaining  
- Wickets in hand  
- Current Run Rate (CRR)  
- Required Run Rate (RRR)  

It converts cricket match data into a **binary classification problem**, where the model outputs win/lose probabilities.

---

## 🧩 Dataset

Source: https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set  

- `matches.csv` → match-level details  
- `deliveries.csv` → ball-by-ball data  

---

## ⚙️ Workflow

### 🧹 Data Preprocessing
- Merged datasets using `match_id`  
- Handled missing values  
- Filtered relevant match conditions  

### 🧮 Feature Engineering
Created important features:
- `runs_left`
- `balls_left`
- `wickets`
- `crr`
- `rrr`

### 🤖 Model Building
- Used Logistic Regression (Scikit-learn)  
- Implemented using a Pipeline  
- Generated win probability predictions  

---

## 📊 Example Output

| Over | Runs | Wickets | Win % |
|------|------|--------|------|
| 1    | 8    | 0      | 45%  |
| 10   | 88   | 3      | 71%  |
| 15   | 123  | 5      | 54%  |

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## 🚀 Future Enhancements

- Build interactive Streamlit web app  
- Use advanced models (Random Forest, XGBoost)  
- Integrate real-time match APIs  

---

## 👨‍💻 Author

**Arnav Singh**

Machine Learning Enthusiast | Aspiring Data Scientist  

📬 **Contact**

Email: itsarnav.singh80@gmail.com  
LinkedIn: https://www.linkedin.com/in/arnav-singh-a87847351  
GitHub: https://github.com/Arnav-Singh-5080

---
