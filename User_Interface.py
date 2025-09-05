import streamlit as st
import pandas as pd
import asyncio
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(__file__))
from Agents import process_csv
import json

# Load API keys from .env and set as environment variables
load_dotenv()

if os.getenv("GROQ_API_KEY_1"):
    os.environ["GROQ_API_KEY_1"] = os.getenv("GROQ_API_KEY_1")
    os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY_1")
if os.getenv("GROQ_API_KEY_2"):
    os.environ["GROQ_API_KEY_2"] = os.getenv("GROQ_API_KEY_2")
if os.getenv("GROQ_API_KEY_3"):
    os.environ["GROQ_API_KEY_3"] = os.getenv("GROQ_API_KEY_3")

st.set_page_config(page_title="Agentic Bias Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Agentic Bias Classifier")
st.markdown("""
<style>
.stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px;}
.stTextInput>div>input {font-size: 18px;}
.stSelectbox>div>div {font-size: 18px;}
</style>
""", unsafe_allow_html=True)

# --- Step 1: Upload CSV and Select Range ---
st.header("Step 1: Upload CSV and Select Range")
uploaded_file = st.file_uploader("Upload CSV for agentic processing", type=["csv"], key="agentic_csv_upload")
if uploaded_file is not None:
    input_csv_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(input_csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")
    df_preview = pd.read_csv(input_csv_path)
    st.dataframe(df_preview.head(10))
    max_range = len(df_preview)
    start_range = st.number_input("Start Range", min_value=0, max_value=max_range-1, value=0)
    end_range = st.number_input("End Range", min_value=start_range+1, max_value=max_range, value=min(1000, max_range))
    if st.button("Proceed with Selected Range"):
        df_range = df_preview.iloc[start_range:end_range].copy()
        temp_csv_path = f"preprocessed_{uploaded_file.name}"
        df_range.to_csv(temp_csv_path, index=False)
        st.success(f"Range selected: Rows {start_range} to {end_range-1}. Preprocessed CSV ready.")
        st.dataframe(df_range.head(10))
        st.session_state["processed_csv_path"] = temp_csv_path
else:
    st.session_state["processed_csv_path"] = None

# --- Load types from types.json for selection ---
with open("types.json", "r", encoding="utf-8") as f:
    types_data = json.load(f)

if "issue_types" not in st.session_state:
    st.session_state["issue_types"] = types_data["issue_types"]
if "bias_types" not in st.session_state:
    st.session_state["bias_types"] = types_data["bias_types"]

# --- Step 2: Agentic Processing ---
st.header("Step 2: Agentic Processing")
good_csv = st.text_input("Good Output CSV", value="good.csv")
issues_csv = st.text_input("Issues Output CSV", value="issues.csv")
log_file = st.text_input("Log File", value="processing_log.txt")

st.write("### Select Issue Types to Flag (add custom below)")
selected_issue_types = st.multiselect("Issue Types", st.session_state["issue_types"], default=st.session_state["issue_types"])
custom_issue_type = st.text_input("Add Custom Issue Type (press Enter to add)", key="custom_issue_type")
if custom_issue_type:
    if custom_issue_type not in st.session_state["issue_types"]:
        st.session_state["issue_types"].append(custom_issue_type)
        selected_issue_types.append(custom_issue_type)

st.write("### Select Bias Types to Flag (add custom below)")
selected_bias_types = st.multiselect("Bias Types", st.session_state["bias_types"], default=st.session_state["bias_types"])
custom_bias_type = st.text_input("Add Custom Bias Type (press Enter to add)", key="custom_bias_type")
if custom_bias_type:
    if custom_bias_type not in st.session_state["bias_types"]:
        st.session_state["bias_types"].append(custom_bias_type)
        selected_bias_types.append(custom_bias_type)

if st.button("Run Agentic Processing"):
    processed_csv_path = st.session_state.get("processed_csv_path", None)
    if processed_csv_path is None or not os.path.exists(processed_csv_path):
        st.error("Please upload a CSV and select a range first.")
    else:
        # Pass selected issue and bias types to process_csv
        async def run_agentic():
            await process_csv(processed_csv_path, good_csv, issues_csv, log_file, columns_to_check=None, issue_types=selected_issue_types, bias_types=selected_bias_types)
        with st.spinner("Processing dataset with agents..."):
            progress_bar = st.progress(0)
            try:
                asyncio.run(run_agentic())
                st.success("Agentic processing completed!")
            except Exception as e:
                st.error(f"Error during processing: {e}")
            progress_bar.progress(100)
            good_df = pd.read_csv(good_csv)
            issues_df = pd.read_csv(issues_csv)
            st.metric("Good Rows", len(good_df))
            st.metric("Flagged Issues", len(issues_df))
            st.write("## Good Rows Sample")
            st.dataframe(good_df.head(10))
            st.write("## Issues Sample")
            st.dataframe(issues_df.head(10))
            # ...existing code...
            # ...existing code...

# --- Step 3: Logs & Download ---
st.header("Step 3: Logs & Download")
if st.button("Show Log File"):
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.error("Log file not found.")

if os.path.exists(good_csv):
    with open(good_csv, "rb") as f:
        st.download_button("Download Good CSV", f, file_name=good_csv, mime="text/csv")
if os.path.exists(issues_csv):
    with open(issues_csv, "rb") as f:
        st.download_button("Download Issues CSV", f, file_name=issues_csv, mime="text/csv")