
import streamlit as st
import pandas as pd
import asyncio
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(__file__))
from Agents import process_csv

# Load API keys from .env and set as environment variables
load_dotenv()
import os

# Set Groq API keys as environment variables
if os.getenv("GROQ_API_KEY_1"):
	os.environ["GROQ_API_KEY_1"] = os.getenv("GROQ_API_KEY_1")
	# Also set OPENAI_API_KEY for compatibility with some clients
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


# Only one file uploader and range selection block is needed above



st.header("Step 2: Agentic Processing")
good_csv = st.text_input("Good Output CSV", value="good.csv")
issues_csv = st.text_input("Issues Output CSV", value="issues.csv")
log_file = st.text_input("Log File", value="processing_log.txt")

if st.button("Run Agentic Processing"):
	processed_csv_path = st.session_state.get("processed_csv_path", None)
	if processed_csv_path is None or not os.path.exists(processed_csv_path):
		st.error("Please upload a CSV and select a range first.")
	else:
		async def run_agentic():
			await process_csv(processed_csv_path, good_csv, issues_csv, log_file)
		with st.spinner("Processing dataset with agents..."):
			asyncio.run(run_agentic())
			st.success("Agentic processing completed!")
			good_df = pd.read_csv(good_csv)
			issues_df = pd.read_csv(issues_csv)
			st.metric("Good Translations", len(good_df))
			st.metric("Flagged Issues", len(issues_df))
			st.write("## Good Translations Sample")
			st.dataframe(good_df.head(10))
			st.write("## Issues Sample")
			st.dataframe(issues_df.head(10))

st.header("Step 3: Logs & Download")
if st.button("Show Log File"):
	with open(log_file, "r", encoding="utf-8") as f:
		st.code(f.read(), language="text")

st.download_button("Download Good CSV", good_csv)
st.download_button("Download Issues CSV", issues_csv)