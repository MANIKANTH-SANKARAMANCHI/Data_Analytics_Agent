# AI Data Analyst Agent (DataPilot)

**Live App:** https://kusuma19072001-ai-data-analyst-agent-app-odnuhd.streamlit.app/

An intelligent Streamlit application that runs an **end-to-end automated analytics pipeline**: upload a tabular dataset, clean and profile the data, explore it with summaries and charts, train baseline machine learning models, and optionally generate a **Gemini-powered narrative report** (with PDF download).

---

## Features

- **Upload** CSV or Excel (`.xlsx`) files
- **Data cleaning** with a transparent cleaning report
- **Data quality** scoring and narrative insights derived from the cleaning report
- **EDA**: summaries, statistical description, correlations
- **Visualizations**: histograms, box plots, distributions, categorical charts, correlation heatmaps, pair plots, feature importance (when applicable)
- **Machine learning**: automatic problem-type detection and model training with comparative metrics
- **AI insights**: structured analytics report via the **Google Gemini API** (optional; requires API key)
- **PDF export** of the AI report with embedded chart images (`fpdf2`)

---

## Workflow

```text
Upload (CSV / XLSX)
        ↓
Data cleaning + quality insights
        ↓
EDA (summary, stats, correlation)
        ↓
Visualizations
        ↓
Model training (optional target column)
        ↓
AI-generated report (optional) → PDF download
```

---

## Tech stack

| Area | Libraries |
|------|-----------|
| App UI | [Streamlit](https://streamlit.io/) |
| Data | pandas, NumPy, openpyxl (Excel) |
| ML | scikit-learn |
| Charts | Matplotlib, Seaborn, Plotly |
| LLM | [google-generativeai](https://ai.google.dev/) (Gemini) |
| Config | python-dotenv |
| PDF | fpdf2 |

---

## Project structure

```text
AI DATA ANALYST AGENT/
├── app.py              # Streamlit entrypoint (DataPilot UI)
├── data_cleaning.py    # Cleaning pipeline + report
├── data_quality.py     # Quality score and insights from cleaning report
├── eda.py              # Summaries, describe, correlation
├── visualization.py    # Plots + chart PNG helpers for PDF
├── model_training.py   # Train/evaluate models, simple ML insights
├── ai_insights.py      # Gemini prompt + report generation
├── utils.py            # Column helpers, target resolution
├── requirements.txt
├── .env.example        # Example environment variables
└── .streamlit/
    └── config.toml     # Streamlit theme/config
```

---

## Prerequisites

- **Python 3.10+** recommended
- A **Google AI (Gemini) API key** if you want AI-generated reports ([Google AI Studio](https://aistudio.google.com/))

---

## Installation

```bash
git clone <your-repo-url>
cd "AI DATA ANALYST AGENT"
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root (you can copy `.env.example`). For AI insights, set:

```env
GEMINI_API_KEY=your_key_here
```

Optional:

```env
GEMINI_MODEL=gemini-1.5-flash-latest
```

If `GEMINI_MODEL` is omitted, the app tries common Gemini model IDs and falls back to the first available model that supports `generateContent`.

---

## Run the application

```bash
streamlit run app.py
```

The app opens in your browser. Upload a dataset, step through the sidebar workflow, pick a target column when you want ML training, then use **Generate AI Insights** when your API key is configured. You can download the full report as **full_ai_report.pdf** when charts and insights are available.

---

## Example use case

Upload something like the Titanic dataset (or any CSV/XLSX with labeled columns). The agent will clean the data, show quality and EDA views, produce charts, train and compare models if you select a target, and summarize findings in business-oriented language via Gemini.

---

## Project goals

- Demonstrate end-to-end **automated data analysis**
- Show a **reproducible ML baseline** pipeline inside an interactive UI
- Combine **deterministic analytics** with **LLM-generated interpretation**

---

## Future improvements

- Broader AutoML / automatic model selection
- Explainability (e.g. SHAP) integrated into the UI
- Richer export formats (HTML / slide decks)
- Cloud deployment (GCP, AWS, etc.)

---

## Collaboration

This project is developed collaboratively by **Kusuma** and **Rajeev**, working together on an AI-assisted data analysis workflow.

---

## Authors

**Kusuma Ch** — Master’s student, Data Science  

**Rajeev** — MCA student, Data Science
