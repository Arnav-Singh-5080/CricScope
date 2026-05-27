<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=d4af37&height=200&section=header&text=CricScope&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Precision%20Match%20Analytics%20for%20Modern%20Cricket&descAlignY=60&descSize=18&descColor=f0d060&animation=fadeIn" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=24&duration=3000&pause=1000&color=D4AF37&center=true&vCenter=true&multiline=true&repeat=true&width=900&height=80&lines=Fintech-grade+Cricket+Intelligence;Official+GSSoC+'26+%26+NSoC+'26+Project" alt="Typing SVG" />

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![License](https://img.shields.io/badge/License-MIT-d4af37?style=for-the-badge)
![GSSoC](https://img.shields.io/badge/GSSoC-2026-ff69b4?style=for-the-badge&logo=github&logoColor=white)
![NSoC](https://img.shields.io/badge/NSoC-2026-brightgreen?style=for-the-badge&logo=github&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-d4af37?style=for-the-badge&logo=git&logoColor=white)

<br/>

![Contributors](https://img.shields.io/github/contributors/Arnav-Singh-5080/cricscope?style=for-the-badge)
![Open Issues](https://img.shields.io/github/issues/Arnav-Singh-5080/cricscope?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/Arnav-Singh-5080/cricscope?style=for-the-badge)

<br/><br/>

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

<br/>

<div align="center">

## What is CricScope?

</div>

**CricScope** is a luxury-grade IPL match intelligence dashboard that computes real-time win probabilities using machine learning — trained on historical ball-by-ball delivery data spanning 2008–2020.

Built with a fintech-inspired dark UI featuring glassmorphism cards, gold gradients, and a premium serif + mono type system. Every design decision was intentional: this is not a student project — it's a production-grade sports analytics product.

> **GSSoC '26 & NSoC 2026 — Project Admin:** [Arnav Singh](https://github.com/Arnav-Singh-5080)

<br/>

---

<br/>

## 📚 Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License](LICENSE)

<br/>

---

<br/>

<div align="center">

## Architecture

</div>

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         CricScope Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   matches.csv ────┐                                                 │
│                   ├──► Merge on match_id ──► Filter: Inning 2       │
│   deliveries.csv ─┘                │                                │
│                                    │                                │
│                      ┌─────────────▼───────────────┐                │
│                      │      Feature Engineering      │               │
│                      │                               │               │
│                      │  current_score  (cumsum)      │               │
│                      │  runs_left      target-score  │               │
│                      │  balls_left     120-ball no.  │               │
│                      │  wickets        10-dismissed  │               │
│                      │  CRR            score/over    │               │
│                      │  RRR            runs*6/balls  │               │
│                      └─────────────┬───────────────┘                │
│                                    │                                │
│                      ┌─────────────▼───────────────┐                │
│                      │       Sklearn Pipeline        │               │
│                      │                               │               │
│                      │  ColumnTransformer            │               │
│                      │    OneHotEncoder              │               │
│                      │      batting_team             │               │
│                      │      bowling_team             │               │
│                      │      city                     │               │
│                      │    passthrough                │               │
│                      │      numeric features         │               │
│                      │                               │               │
│                      │  LogisticRegression           │               │
│                      │    max_iter = 1000            │               │
│                      └─────────────┬───────────────┘                │
│                                    │                                │
│                      ┌─────────────▼───────────────┐                │
│                      │ predict_proba() → [0, 1]    │                │
│                      │ Confidence: High/Mod/Close  │                │
│                      └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```


<div align="center">

##  Model Evaluation Metrics

</div>

The prediction engine uses a **Logistic Regression classifier**
trained on IPL ball-by-ball match data spanning IPL seasons from 2008–2020.

### Dataset Split

| Split Type | Ratio |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |

---

### Model Highlights

- Real-time win probability prediction
- Ball-by-ball dynamic recalculation
- Match-state feature engineering
- OneHotEncoded categorical preprocessing
- Optimized scikit-learn pipeline
- Context-aware chase prediction logic

---

> Detailed evaluation metrics and expanded cross-validation results will be added in future model benchmarking updates.
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
| Styling | Glassmorphism + Premium CSS | Luxury UI/UX |
| ML Pipeline | scikit-learn | Preprocessing + Logistic Regression |
| Data | Pandas + NumPy | Feature engineering |
| Deployment | Streamlit Cloud | Live hosting |

</div>

<br/>

---

<br/>

<div align="center">

## Project Structure

</div>


## Screenshots

### Dashboard

![Dashboard](assets/dashboard.png)

### Win Probability Prediction

![Win Probability Prediction](assets/win-probability-prediction.png)

### Analytics

![Analytics](assets/analytics.png)


## Demo

![Demo](assets/cricscope.gif)

```bash
cricscope/
│
├── assets/
│   ├── dashboard.png
│   ├── prediction-page.png
│   ├── analytics.png
│   
├── app.py
├── matches.csv
├── deliveries.csv
├── requirements.txt
└── README.md
└── demo_.gif
```

<br/>

---

<br/>

<div align="center">

## Getting Started

</div>

### Prerequisites

- Python 3.9+
- IPL Dataset

Dataset:
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

---

## 1 — Clone Repository

```bash
git clone https://github.com/Arnav-Singh-5080/cricscope.git
cd cricscope
```

---

## 2 — Create Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4 — Add Dataset

```text
cricscope/
├── app.py
├── matches.csv
└── deliveries.csv
```

---

## 5 — Run Application

```bash
streamlit run app.py
```

Visit:

```text
http://localhost:8501
```

<br/>

---

<br/>

<div align="center">

## Open Source Programs 2026

<img src="https://img.shields.io/badge/GSSoC%20'26%20%26%20NSoC%202026-d4af37?style=for-the-badge&logo=github&logoColor=white" />

</div>

<br/>

CricScope is an officially selected project under:

- **GirlScript Summer of Code 2026 (GSSoC '26)**
- **Nexus Summer of Code 2026 (NSoC 2026)**

Contributors are evaluated on:

- Code quality
- UI consistency
- ML innovation
- Documentation quality
- Feature implementation

This repository welcomes:

- Beginners
- Open-source contributors
- UI/UX developers
- ML engineers
- Competitive programmers

<br/>

---

<br/>

<div align="center">

## Why Contribute?

</div>

- Beginner-friendly contribution environment
- Real-world ML engineering exposure
- Production-grade UI/UX experience
- Active mentoring & PR reviews
- Portfolio-worthy open-source project
- Industry-style collaboration workflow

<br/>

---

<br/>

<div align="center">

## Contribution Guide

</div>

### How to Contribute

```bash
# 1. Fork the repository

# 2. Clone your fork
git clone https://github.com/<your-username>/cricscope.git

# 3. Move into project
cd cricscope

# 4. Create feature branch
git checkout -b feature/your-feature-name

# 5. Make changes

# 6. Commit
git add .
git commit -m "feat: describe your changes"

# 7. Push
git push origin feature/your-feature-name

# 8. Open Pull Request
```

---

### Contribution Guidelines

- Maintain the premium dark luxury aesthetic
- Follow the existing project structure
- Keep pull requests focused
- Test locally before opening PR
- Write meaningful commit messages
- Respect UI consistency

---
## 🔐 Security Policy

Please review the [Security Policy](SECURITY.md) for vulnerability reporting guidelines and supported versions.

---

## 🤝 Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing to the project.


### Good First Issues

| Area | Task | Difficulty |
|:-----|:-----|:----------:|
| UI | Animated win probability graph | Medium |
| UI | Mobile responsiveness | Medium |
| UI | Team stat pills | Easy |
| ML | IPL 2021–2024 integration | Easy |
| ML | SHAP interaction visualizations | Medium |
| ML | Cross-validation metrics | Medium |
| Feature | Match report PDF export | Hard |
| Feature | Head-to-head analytics | Medium |
| Docs | Add screenshots & GIF demo | Easy |

<br/>

---

<br/>

<div align="center">

## Project Admin

### **Arnav Singh**

Project Admin — GSSoC '26 & NSoC 2026

<br/>

[![Email](https://img.shields.io/badge/Email-itsarnav.singh80%40gmail.com-d4af37?style=for-the-badge&logo=gmail&logoColor=white)](mailto:itsarnav.singh80@gmail.com)

&nbsp;

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arnav-singh-a87847351)

&nbsp;

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Arnav-Singh-5080)

</div>

<br/>

---

<br/>



<div align="center">

## Support the Project

If this project helped you, consider giving it a ⭐ on GitHub.

It helps more contributors discover the project and motivates future development.

<br/><br/>


<img src="https://capsule-render.vercel.app/api?type=waving&color=d4af37&height=100&section=footer&animation=fadeIn" width="100%"/>

</div>
