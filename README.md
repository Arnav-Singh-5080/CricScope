<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=d4af37&height=200&section=header&text=CricScope&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Precision%20Match%20Analytics%20for%20Modern%20Cricket&descAlignY=60&descSize=18&descColor=f0d060&animation=fadeIn" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=24&duration=3000&pause=1000&color=D4AF37&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=80&lines=Fintech-grade+Cricket+Intelligence;Official+NSoC+%2726+Project" alt="Typing SVG" />

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![License](https://img.shields.io/badge/License-MIT-d4af37?style=for-the-badge)
![NSoC](https://img.shields.io/badge/NSoC-2026-brightgreen?style=for-the-badge&logo=github&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-d4af37?style=for-the-badge&logo=git&logoColor=white)

<br/>

<a href="https://github.com/Arnav-Singh-5080/cricscope/stargazers">
  <img src="https://img.shields.io/github/stars/Arnav-Singh-5080/cricscope?style=social" />
</a>
&nbsp;
<a href="https://github.com/Arnav-Singh-5080/cricscope/network/members">
  <img src="https://img.shields.io/github/forks/Arnav-Singh-5080/cricscope?style=social" />
</a>
&nbsp;
<a href="https://github.com/Arnav-Singh-5080/cricscope/issues">
  <img src="https://img.shields.io/github/issues/Arnav-Singh-5080/cricscope?style=social" />
</a>

</div>

<br/>

<br/>

<div align="center">

## Live Demo

<a href="https://cricscope-live.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/Launch%20App-CricScope-d4af37?style=for-the-badge&logo=streamlit&logoColor=white" />
</a>

</div>

<br/>

---
## What is CricScope?

**CricScope** is a luxury-grade IPL match intelligence dashboard that computes real-time win probabilities using machine learning — trained on historical ball-by-ball delivery data spanning 2008–2020.

Built with a fintech-inspired dark UI featuring glassmorphism cards, gold gradients, and a premium serif + mono type system. Every design decision was intentional: this is not a student project — it's a production-grade sports analytics product.

**NSoC 2026 — Project Admin:** [Arnav Singh](https://github.com/Arnav-Singh-5080)

<br/>


## 📌 Overview

CricScope is designed to explore, analyze, and visualize cricket data to uncover patterns, player performance trends, and match insights.

Whether you're a cricket enthusiast, data analyst, or developer, this project demonstrates how data science can enhance sports understanding.

<br/>

---

<br/>

## 🔄 Workflow

```text
Data Collection → Data Cleaning → EDA → Visualization → Feature Engineering → Modeling → Evaluation
```

<div align="center">

## Architecture

</div>

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CricScope Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   matches.csv ────┐                                                   │
│                   ├──► Merge on match_id ──► Filter: Inning 2        │
│   deliveries.csv ─┘                │                                  │
│                                    │                                  │
│                      ┌─────────────▼───────────────┐                 │
│                      │      Feature Engineering      │                │
│                      │                               │                │
│                      │  current_score  (cumsum)      │                │
│                      │  runs_left      target - score│                │
│                      │  balls_left     120 - ball no.│                │
│                      │  wickets        10 - dismissed│                │
│                      │  CRR            score / over  │                │
│                      │  RRR            runs*6 / balls│                │
│                      └─────────────┬───────────────┘                 │
│                                    │                                  │
│                      ┌─────────────▼───────────────┐                 │
│                      │      Sklearn Pipeline         │                │
│                      │                               │                │
│                      │  ColumnTransformer            │                │
│                      │    OneHotEncoder               │                │
│                      │      batting_team              │                │
│                      │      bowling_team              │                │
│                      │      city                      │                │
│                      │    passthrough                 │                │
│                      │      numeric features          │                │
│                      │                               │                │
│                      │  LogisticRegression           │                │
│                      │    max_iter = 1000             │                │
│                      └─────────────┬───────────────┘                 │
│                                    │                                  │
│                      ┌─────────────▼───────────────┐                 │
│                      │   predict_proba() → [0, 1]   │                │
│                      │   + Confidence: High/Mod/Close│                │
│                      └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

<br/>

---

<br/>

<div align="center">

## Tech Stack

<img src="https://skillicons.dev/icons?i=python,git,github,vscode&theme=dark" />

<br/><br/>

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| Frontend | Streamlit + Custom CSS | UI rendering & layout |
| Styling | Glassmorphism, CSS blend modes | Premium visual design |
| ML Pipeline | scikit-learn | Preprocessing + LogReg |
| Data | Pandas, NumPy | Feature engineering |

</div>

<br/>

---

<br/>

<div align="center">

## Features

</div>

<table>
<tr>
<td width="50%" valign="top">

**Prediction Engine**

- Real-time win probability on every input change
- Computes CRR, RRR, balls remaining, wickets in hand
- Confidence scoring: High / Moderate / Close
- Dual prediction cards — batting vs bowling team

</td>
<td width="50%" valign="top">

**Design System**

- Glassmorphism cards with `backdrop-filter: blur`
- Gold gradient typography (`#d4af37` → `#f0d060`)
- `mix-blend-mode: screen` for clean logo rendering
- Cormorant Garamond + DM Sans + DM Mono

</td>
</tr>
<tr>
<td width="50%" valign="top">

**All 8 IPL Teams**

- CSK · MI · RCB · KKR · DC · RR · SRH · PBKS
- Team-colored glow effects per brand identity
- Circular logo containers with dark base layer

</td>
<td width="50%" valign="top">

**Sidebar & Navigation**

- Premium profile card with clickable contact links
- Built-by section with avatar initials, role label
- Navigation between Dashboard and Analysis pages

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## Project Structure

</div>

```
cricscope/
│
├── app.py                  # Main Streamlit app — UI + ML inference
├── matches.csv             # IPL match-level data  (2008–2020)
├── deliveries.csv          # Ball-by-ball delivery data
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

<br/>

---

<br/>

<div align="center">

## Getting Started

</div>

### Prerequisites

- Python 3.9+
- IPL dataset: [Kaggle — IPL Complete Dataset 2008–2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

### 1 — Clone

```bash
git clone https://github.com/Arnav-Singh-5080/cricscope.git
cd cricscope
```

### 2 — Virtual environment

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy scikit-learn
```

### 4 — Dataset

Place both CSV files in the project root alongside `app.py`:

```
cricscope/
├── app.py
├── matches.csv       ← place here
└── deliveries.csv    ← place here
```

### 5 — Run

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501)

<br/>

---

<br/>

<div align="center">

## NSoC 2026

<img src="https://img.shields.io/badge/Open%20Source%20Contributions%202026-d4af37?style=for-the-badge&logo=github&logoColor=white" />

<br/><br/>

CricScope is an **officially selected project under NSoC 2026** (Nexus Spring of Code).
Contributors are evaluated on code quality, UI consistency, and ML improvements.

</div>

<br/>

### How to contribute

```bash
# 1. Fork this repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your-username>/cricscope.git
cd cricscope

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes, then commit
git add .
git commit -m "feat: describe your change clearly"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request to the main branch of this repo
```

### Contribution guidelines

- Follow the existing code style — inline styles for Streamlit HTML, clear docstrings for functions
- Do **not** modify the ML pipeline unless your PR is specifically about model improvement
- All UI changes must maintain the dark gold luxury aesthetic — no light backgrounds, no generic fonts
- Test locally with both CSV files present before opening a PR
- Keep PRs focused — one feature or fix per PR

### Good first issues

| Area | Task | Difficulty |
|:-----|:-----|:----------:|
| UI | Over-by-over probability animated line chart | Medium |
| UI | Mobile-responsive sidebar and card layout | Medium |
| UI | Add team win-rate stat pills to dashboard | Easy |
| ML | Integrate IPL 2021–2024 dataset | Easy |
| ML | Add Random Forest / XGBoost model toggle | Hard |
| ML | Cross-validation accuracy display | Medium |
| Feature | Historical head-to-head team stats panel | Medium |
| Feature | Downloadable match report PDF | Hard |
| Docs | Add screenshots/GIF demo to README | Easy |

<br/>
---

## 📈 Future Improvements

* 🔮 Advanced predictive models (deep learning)
* 🌐 Web dashboard (Streamlit / React)
* 📡 Real-time data integration
* 📊 Interactive dashboards

---
## 🐛 Issues

Found a bug or have suggestions?
Open an issue with proper details.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you like this project:

* Star ⭐ the repository
* Share it with others

---

## 🙌 Acknowledgements

Inspired by the intersection of sports and data science.
---

---

<br/>

<div align="center">


## Project Admin



**Arnav Singh** — Project Admin, NSoC 2026

[![Email](https://img.shields.io/badge/Email-itsarnav.singh80%40gmail.com-d4af37?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsarnav.singh80@gmail.com)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arnav-singh-a87847351)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Arnav-Singh-5080)

<br/>

*If this project helped you, consider leaving a star. It helps more contributors find it.*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=d4af37&height=100&section=footer&animation=fadeIn" width="100%"/>

</div>
