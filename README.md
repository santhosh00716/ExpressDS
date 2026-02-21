# ExpressDS - Automated Data Science Platform

An automated Data Science platform where users upload a CSV, and the AI cleans data, performs EDA, trains the best ML model via AutoML, and lets you chat with your data.

## Tech Stack

- **Python** - Core language
- **Streamlit** - Web frontend
- **PyCaret** - AutoML
- **LangChain + Google Gemini** - Agentic chat with data
- **Pandas** - Data processing

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies (PyCaret has specific version requirements)
pip install -r requirements.txt

# If you hit dependency conflicts, try:
# pip install streamlit pandas openai langchain-openai langchain-experimental
# pip install pycaret  # may need compatible numpy/pandas/matplotlib
```

**Gemini API Key** (for AI Chat feature): Set `GEMINI_API_KEY` in `.env` or enter in the app sidebar. Get a free key at https://aistudio.google.com/apikey

## Run

```bash
streamlit run main.py
```

## Features

1. **Data Upload** - Upload CSV files
2. **Data Cleaning** - Auto-fix missing values, duplicates, data types
3. **Data Health Report** - Summary of cleaning operations
4. **Auto-EDA** - Exploratory data analysis, correlations
5. **Model Training** - PyCaret AutoML (classification/regression)
6. **AI Chatbot** - Natural language queries on your data (Gemini API key required)

## Project Structure

```
ExpressDS-02/
├── main.py              # Streamlit app
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── data_engine/     # Data cleaning
    ├── automl/          # PyCaret AutoML
    ├── chat_agent/      # LangChain + Gemini agent
    └── validation/      # GIGO protection
```
