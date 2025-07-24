# 🚀 Startup Pitch Deck Evaluator using NLP

This project builds an AI-powered evaluator that reads real-world startup pitch decks and scores them on key dimensions like Problem Clarity, Market Potential, Traction, Team Experience, and more — helping identify the most investable startups.

---

## 📌 Objective
To analyze 5–10 real startup pitch decks and extract meaningful signals using NLP, generating a final quality score and ranking for each deck.

---

## 📁 Dataset
Pitch decks in PDF or slide format provided via Google Drive. Each deck may include:
- Problem Statement
- Solution
- Market Size (TAM/SAM/SOM)
- Traction Graphs
- Business Model
- Team Overview
- Funding Ask

---

## 🧠 NLP Pipeline

### 1. **Text Extraction & Parsing**
- Tools: `PyMuPDF`, `pdfplumber`, `pdfminer`
- Clean raw text, split into logical sections (e.g., Problem, Market, Team)

### 2. **Scoring Dimensions**
Each deck is scored on the following (0–10 scale):
| Dimension         | Example Indicators                                 |
|------------------|-----------------------------------------------------|
| Problem Clarity  | Clearly defined pain point                         |
| Market Potential | Large TAM/SAM/SOM, real demand                     |
| Traction Strength| Growth %, revenue, user stats                      |
| Team Experience  | Domain expertise, previous startups, skills        |
| Business Model   | Clear revenue model (SaaS, commission, etc.)       |
| Vision/Moat      | Defensibility, data/IP advantage                   |
| Confidence       | Persuasive, confident tone                         |

- Final Score = Normalized total (out of 100)

---

## 📊 Output

- 📄 `CSV/XLSX`: Deck name + all scores
- 📈 Visualizations:
  - Radar Chart per Deck
  - Score Distribution Histogram
  - Correlation Heatmap
- 🏆 Ranked Table: Top 3 / Bottom 3 decks
- 💡 Insight: 1-line investability comment per deck

---

## 🛠️ Tools & Libraries

- **Text Extraction**: PyMuPDF / pdfplumber
- **Preprocessing**: Python (re, nltk, spaCy)
- **Scoring Logic**: Custom NLP logic or LLMs (Hugging Face, OpenAI)
- **Visualization**: matplotlib / seaborn / plotly
- **Dashboard (optional)**: Streamlit / Dash

---

## 🌟 Bonus Features
- 🔍 Auto-classify decks (e.g., FinTech, SaaS, HealthTech)
- 🧾 LLM-based summaries (4 bullet points per deck)

---

## 📂 Deliverables

- ✅ Python Code Notebook
- ✅ Final Scores Report (CSV + PDF)
- ✅ README (This file)
- ✅ Visualizations (charts/images)
