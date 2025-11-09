import streamlit as st
import tempfile
import json
import os
from data_processor import (
    initialize_llm,
    extract_raw_json_from_pdf,
    extract_specific_health_data,
    get_health_score_dynamic_langchain
)
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

# --- Configuration ---
# Use st.secrets in a deployed Streamlit app, os.getenv for local testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

st.set_page_config(
    page_title="HealthSync: AI-Powered Medical Report Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar & API Key Check ---

with st.sidebar:
    st.title("Settings")
    
    # Check for API Key
    if not OPENAI_API_KEY:
        st.warning("🚨 OpenAI API Key not found. Please set the `OPENAI_API_KEY` in your environment variables or `st.secrets`.")
        st.stop()
    
    st.success("✅ OpenAI API Key Loaded.")
    
    # Model selection (optional)
    model = st.selectbox(
        "Select LLM Model",
        options=["gpt-4o", "gpt-4-turbo"],
        index=0
    )
    st.info("Using a powerful model like gpt-4o ensures high-quality extraction and clinical reasoning.")
    
    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
        1. **Upload:** Upload a PDF medical report.
        2. **Extract:** An LLM extracts all key-value pairs into a raw JSON.
        3. **Analyze:** The LLM interprets the data to calculate a **Health Score** and provide a detailed breakdown.
    """)

# --- Main App Interface ---

st.title("HealthSync: AI Medical Report Analyzer 🤖")
st.markdown("Upload a PDF lab report to get a structured data summary and a clinical health score analysis.")

uploaded_file = st.file_uploader("Upload your PDF Medical Report", type=["pdf"])

if uploaded_file is not None:
    # Use a temporary file to save the uploaded PDF for the UnstructuredPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    try:
        # 1. Initialize LLM
        llm = initialize_llm(OPENAI_API_KEY, model_name=model)
        
        st.header("Step 1: Raw Data Extraction")
        st.markdown("---")
        
        with st.spinner("🧠 Extracting structured data from PDF (using Unstructured and LLM)..."):
            # 2. Extract Raw JSON
            raw_json_model = extract_raw_json_from_pdf(temp_file_path, llm)
            
            if raw_json_model:
                raw_json_data = raw_json_model.model_dump()
                # Prepare JSON string for download
                raw_json_string = json.dumps(raw_json_data, indent=2)
                
                st.success("✅ Raw Data Extraction Complete.")
                
                # Download button for Raw JSON (Step 1 UI Enhancement)
                col_dl1, col_space1 = st.columns([2, 5])
                with col_dl1:
                    st.download_button(
                        label="Download Raw JSON 💾",
                        data=raw_json_string.encode('utf-8'),
                        file_name="raw_extracted_data.json",
                        mime="application/json",
                        type="secondary"
                    )
                
                # Display extracted data
                with st.expander("Show Raw Extracted JSON", expanded=False):
                    st.json(raw_json_data)
                
                st.header("Step 2: Health Score Analysis")
                st.markdown("---")
                
                # 3. Get Health Score
                with st.spinner("🔬 Analyzing lab results and calculating Health Score..."):
                    health_score_raw_output = get_health_score_dynamic_langchain(raw_json_data, llm)
                    
                    try:
                        # Find the JSON block and parse it
                        start_index = health_score_raw_output.find('{')
                        end_index = health_score_raw_output.rfind('}') + 1
                        
                        if start_index == -1 or end_index == 0:
                            raise ValueError("Could not find valid JSON object in LLM response.")

                        analysis_json_data = health_score_raw_output[start_index:end_index]
                        health_score_json = json.loads(analysis_json_data)
                        
                        score = health_score_json.get("score", 0)
                        summary = health_score_json.get("summary_reasoning", "No summary provided.")
                        breakdown = health_score_json.get("detailed_breakdown", [])
                        
                        # Display the main results
                        col1, col2, col_dl2 = st.columns([1, 2.5, 1.5]) # Add a column for the download button
                        with col1:
                            st.metric(label="Health Score (Out of 100)", value=f"**{score}**")
                        with col2:
                            st.subheader("Overall Summary")
                            st.info(summary)
                        with col_dl2:
                            st.markdown(" ") # Spacer
                            st.markdown(" ") # Spacer
                            # Download button for Health Analysis (Step 2 UI Enhancement)
                            st.download_button(
                                label="Download Analysis JSON ⬇️",
                                data=analysis_json_data.encode('utf-8'),
                                file_name="health_analysis_report.json",
                                mime="application/json",
                                type="primary"
                            )

                        # Display detailed breakdown
                        st.subheader("Detailed Parameter Breakdown")
                        st.dataframe(
                            data=breakdown,
                            column_order=["parameter", "value", "status", "analysis"],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Section for the second notebook task (Graph data extraction)
                        st.header("Step 3: Graph Data Extraction (Optional)")
                        st.markdown("---")
                        with st.spinner("📊 Extracting specific parameters for time series visualization..."):
                            specific_health_data_model = extract_specific_health_data(raw_json_data, llm)
                            if specific_health_data_model:
                                graph_data = specific_health_data_model.model_dump()
                                st.success("✅ Specific Parameters Extracted.")
                                with st.expander("Show Graph-Ready Data", expanded=False):
                                    st.json(graph_data)
                                    st.markdown(f"**Report Date:** `{graph_data['reportDate']}`")
                                    st.markdown(f"**Parameters:** `{graph_data['healthParameters']}`")

                            else:
                                st.error("Failed to extract specific parameters for graphing.")

                    except json.JSONDecodeError:
                        st.error("Error: Could not parse the LLM's health score output. Displaying raw output:")
                        st.code(health_score_raw_output, language='json')
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {e}")

            else:
                st.error("🔴 Failed to extract structured JSON from the PDF.")
        
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
    except Exception as e:
        st.error(f"An unknown error occurred: {e}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

else:
    st.info("👆 Please upload a PDF file to begin the health analysis.")