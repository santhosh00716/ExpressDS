"""
ExpressDS - Automated Data Science Platform
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_engine.cleaner import clean_data, DataHealthReport
from src.automl.engine import run_express_ml
from src.chat_agent.agent import create_data_agent, chat_with_data
from src.validation.validator import validate_dataset, ValidationResult

# Page config - modern look
st.set_page_config(
    page_title="ExpressDS | Automated Data Science",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }

    .metric-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    if "health_report" not in st.session_state:
        st.session_state.health_report = None
    if "ml_result" not in st.session_state:
        st.session_state.ml_result = None
    if "data_agent" not in st.session_state:
        st.session_state.data_agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("## 📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    gemini_key = st.text_input(
        "Gemini API Key (for AI Chat)",
        type="password",
        placeholder="AIza...",
        help="Get a free key at aistudio.google.com/apikey",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.markdown("---")
    st.markdown("### 📋 Steps")
    st.markdown("1. Upload CSV")
    st.markdown("2. Preview & clean data")
    st.markdown("3. Run Auto-EDA")
    st.markdown("4. Train models")
    st.markdown("5. Chat with data")

# Load and clean data
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df_raw

        # Validation (GIGO protection)
        validation = validate_dataset(df_raw, target_column=None, min_rows=10)
        if not validation.is_valid:
            st.error("**Data validation failed:**")
            for err in validation.errors:
                st.error(f"• {err}")
            if validation.warnings:
                for w in validation.warnings:
                    st.warning(w)
            st.stop()

        # Clean data
        df_clean, health_report = clean_data(df_raw)
        st.session_state.df_clean = df_clean
        st.session_state.health_report = health_report

        # Warnings from validation
        validation_full = validate_dataset(df_raw, target_column=None)
        for w in validation_full.warnings:
            st.sidebar.warning(w)

    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
else:
    st.info("👈 Upload a CSV file to get started.")
    st.stop()

# Main content
st.markdown('<div class="main-header"><h1>📊 ExpressDS</h1><p>Automated Data Science Platform — Clean, Explore, Model, Chat</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Preview",
    "📈 Auto-EDA",
    "🤖 Model Training",
    "💬 AI Chatbot",
    "📑 Data Health Report",
])

with tab1:
    st.subheader("Cleaned Data Preview")
    st.dataframe(st.session_state.df_clean, use_container_width=True, height=400)
    st.caption(f"Shape: {st.session_state.df_clean.shape[0]} rows × {st.session_state.df_clean.shape[1]} columns")

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.dataframe(st.session_state.df_clean.describe(), use_container_width=True)
    st.markdown("#### Numeric distributions")
    numeric_cols = st.session_state.df_clean.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        selected = st.selectbox("Select column for histogram", numeric_cols)
        if selected:
            st.bar_chart(st.session_state.df_clean[selected].value_counts().head(20))
    st.markdown("#### Correlation (numeric columns)")
    if len(numeric_cols) >= 2:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = st.session_state.df_clean[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("AutoML Model Training")
    target_col = st.selectbox(
        "Select target column",
        st.session_state.df_clean.columns.tolist(),
        key="target_select",
    )

    # Validate before ML
    ml_validation = validate_dataset(st.session_state.df_clean, target_column=target_col)
    if not ml_validation.is_valid:
        st.error("**Cannot train model:**")
        for err in ml_validation.errors:
            st.error(f"• {err}")
    else:
        if st.button("Train Models"):
            with st.spinner("Training models with PyCaret..."):
                try:
                    result = run_express_ml(
                        st.session_state.df_clean,
                        target_column=target_col,
                        top_n=3,
                    )
                    st.session_state.ml_result = result
                except Exception as ex:
                    st.session_state.ml_result = {"error": str(ex)}

        if st.session_state.ml_result:
            res = st.session_state.ml_result
            if res.get("error"):
                st.error(f"Training error: {res['error']}")
            else:
                st.success(f"Problem type: **{res['problem_type']}**")
                if res.get("top_models"):
                    st.markdown("#### Top Models")
                    for m in res["top_models"]:
                        st.markdown(f"- **{m.get('model', 'N/A')}**")
                if res.get("feature_importance_plot"):
                    import base64
                    from PIL import Image
                    import io
                    img_data = base64.b64decode(res["feature_importance_plot"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
                elif res.get("feature_importance_error"):
                    st.info("Feature importance not available for this model type.")

with tab4:
    st.subheader("Chat with Data")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.warning("Enter your Gemini API Key in the sidebar to use the AI chatbot.")
    else:
        if st.session_state.data_agent is None:
            try:
                agent = create_data_agent(st.session_state.df_clean)
                st.session_state.data_agent = agent
            except Exception as ex:
                st.error(f"Failed to create agent: {ex}")
                agent = None
        else:
            agent = st.session_state.data_agent

        if agent:
            question = st.chat_input("Ask about your data (e.g., 'What is the average of column X?')")
            if question:
                response = chat_with_data(agent, question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

with tab5:
    st.subheader("Data Health Report")
    report = st.session_state.health_report
    if report:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Original rows", report.original_rows)
        with c2:
            st.metric("Final rows", report.final_rows)
        with c3:
            st.metric("Duplicates removed", report.duplicates_removed)
        st.markdown("#### Missing values filled")
        if report.missing_values_filled:
            st.json(report.missing_values_filled)
        else:
            st.info("No missing values to impute.")
        st.markdown("#### Imputation methods")
        if report.columns_imputed:
            st.json(report.columns_imputed)
        if report.type_conversions:
            st.markdown("#### Type conversions")
            st.json({k: str(v) for k, v in report.type_conversions.items()})
