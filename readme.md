# ⚕️ HealthSync: AI Medical Report Analyzer

## 📋 Overview

HealthSync is an intelligent AI-powered application that transforms PDF medical lab reports into actionable insights. Built with Streamlit and LangChain, it leverages Large Language Models (LLMs) to provide comprehensive medical data analysis.

### Core Features

1. **Raw Data Extraction**: Extracts all key-value data from PDF reports into structured JSON format
2. **Health Score Analysis**: Calculates clinical health scores (0-100) with detailed breakdowns and summaries
3. **Graph Data Extraction**: Extracts key parameters for time-series visualization and trend analysis

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd HealthSync
```

### 2. Environment Setup

#### A. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

#### B. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. API Key Configuration

#### A. Create `.env` File

Create a new file named `.env` in the root directory:

```bash
touch .env  # macOS/Linux
type nul > .env  # Windows
```

#### B. Add Your OpenAI API Key

Open `.env` and add:

```ini
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> **Note**: For production deployment on Streamlit Cloud, use Streamlit Secrets instead of `.env` file.

---

## 💻 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 🛠️ How to Use HealthSync

### Step-by-Step Guide

1. **Verify API Key**: Check the sidebar for "✅ OpenAI API Key Loaded" confirmation
2. **Select Model**: Choose your preferred LLM model (default: `gpt-4o`)
3. **Upload Report**: Use the file uploader to select your PDF medical report
4. **Processing**: The app automatically executes three analysis steps:
   - **Step 1**: Raw Data Extraction → Displays structured JSON
   - **Step 2**: Health Score Analysis → Shows score, summary, and detailed breakdown
   - **Step 3**: Graph Data Extraction → Extracts visualization-ready parameters
5. **Review & Download**: Examine results and download JSON outputs

---

## 📁 Project Structure

```
HealthSync/
│
├── app.py                  # Main Streamlit application
├── data_processor.py       # LangChain logic & LLM integration
├── requirements.txt        # Python dependencies
├── .env                    # API key configuration (create this)
├── BP1(INPUT)              # Blood Pressure Medical Report
├── Urine(INPUT)            # Urine Medical Report 
└── README.md              # Project documentation
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, file upload handling, temporary file management |
| `data_processor.py` | Core LangChain logic, Pydantic models, LLM functions |
| `requirements.txt` | All required Python packages |
| `.env` | Stores `OPENAI_API_KEY` (git-ignored) |
| `BP1` | Blood Pressure Medical report of an Patient X to Test the Model|
| `Urine` | Urine Medical report of an Paient Y to Test the Model |
---

## 📦 Dependencies

Key libraries used in this project:

- `streamlit` - Web application framework
- `langchain-openai` - LLM integration
- `langchain-community` - Community tools and loaders
- `unstructured[pdf]` - PDF processing and extraction
- `python-dotenv` - Environment variable management
- `pydantic` - Data validation and parsing

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🛑 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **"OpenAI API Key not found"** | Ensure `.env` file exists in root directory with correct `OPENAI_API_KEY` format |
| **ImportError: No module named 'langchain_community'** | Run `pip install -r requirements.txt` to install all dependencies |
| **"Failed to extract structured JSON"** | PDF may be image-heavy or have complex formatting. Consider OCR pre-processing |
| **json.JSONDecodeError** | LLM output parsing failed. Try switching to `gpt-4o` model or adjust temperature |
| **Streamlit won't start** | Verify Python version (3.8+) and that virtual environment is activated |

### Debug Mode

Enable verbose logging by adding to your `.env`:
```ini
DEBUG=True
```

---

## 🔒 Security & Privacy

- **API Keys**: Never commit `.env` file to version control
- **Patient Data**: Ensure compliance with HIPAA/GDPR when processing medical reports
- **Data Storage**: No data is stored permanently; temporary files are cleaned after processing

---

## 🚀 Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub (exclude `.env`)
2. Connect repository to Streamlit Cloud
3. Add `OPENAI_API_KEY` to Streamlit Secrets:
   - Go to App Settings → Secrets
   - Add: `OPENAI_API_KEY = "sk-..."`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain community for excellent tools
- Streamlit for the amazing framework

---

## ⚠️ Disclaimer

This application is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.

---

Made with ❤️ by the HealthSync Team
