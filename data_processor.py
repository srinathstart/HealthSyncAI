import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Pydantic Models for Output Parsing ---

class DynamicJSONModel(BaseModel):
    """Model for initial unstructured extraction into a dictionary."""
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary containing dynamically extracted key-value pairs from the document"
    )

class ExtractedHealthData(BaseModel):
    """Model for extracting specific health parameters and date for charting."""
    reportDate: str = Field(..., description="The date of the medical report in YYYY-MM-DD format.")
    healthParameters: Dict[str, Any] = Field(
        ...,
        description="A dictionary of health-related key-value pairs."
    )

# --- Core Functions ---

def initialize_llm(api_key: str, model_name: str = "gpt-4o", temperature: float = 0.7):
    """Initializes the ChatOpenAI model."""
    if not api_key:
        raise ValueError("OpenAI API key is required.")
    return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)

def extract_raw_json_from_pdf(file_path: str, llm: ChatOpenAI) -> Optional[DynamicJSONModel]:
    """
    Loads a PDF, extracts text, and uses the LLM to convert it to a structured JSON object.
    """
    try:
        # 1. Load the PDF
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="ocr_only",
        )
        docs = loader.load()
        document_text = " ".join([doc.page_content for doc in docs])

        print(f"Extracted document text length: {len(document_text)}")
        if len(document_text) < 100:
            print(f"Document Text Snippet: {document_text[:50]}...")

        # 2. Setup LangChain components
        parser = PydanticOutputParser(pydantic_object=DynamicJSONModel)

        prompt_template = '''
        You are a highly skilled data extraction and conversion assistant. Your task is to extract key-value pairs from the provided medical report and structure them into a JSON object.

        **Input:**
        {context}

        **Instructions:**
        1.  **Identify Key-Value Pairs:** Carefully read through the medical report and identify all relevant information that can be represented as a key-value pair.
        2.  **Naming Conventions:**
            * Keys should be in **camelCase**.
            * Keys should be descriptive and concise (e.g., `patientName`, `dateOfBirth`, `diagnosis`, `medications`).
            * Data Types: Ensure the values are of the correct data type (e.g., numbers, strings, or booleans). If a field contains multiple items (like a list of medications), use a JSON array.
        3.  **Structure:** The final output must be a single, valid JSON object. **Crucially, output ONLY the JSON object and nothing else. Do not include any introductory or concluding text, explanations, or conversational phrases outside of the JSON block.**

        {format_instructions}
        '''

        prompt = PromptTemplate(
            input_variables=["context"],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = {"context": RunnablePassthrough()} | prompt | llm | parser

        # 3. Invoke the chain
        result = chain.invoke({"context": document_text})
        return result

    except Exception as e:
        print(f"Error during raw JSON extraction: {e}")
        return None

def extract_specific_health_data(json_data: Dict[str, Any], llm: ChatOpenAI) -> Optional[ExtractedHealthData]:
    """
    Analyzes the extracted JSON and returns a smaller JSON with specific health parameters.
    """
    try:
        json_string = json.dumps(json_data, indent=2)

        # 1. Setup LangChain components
        parser = PydanticOutputParser(pydantic_object=ExtractedHealthData)

        prompt_template = '''
        You are a highly skilled medical data extraction assistant. Your task is to analyze a JSON object containing medical report data and extract the report date and exactly four compulsory health parameters based on the report type.

        **Input JSON:**
        {context}

        **Instructions:**
        1. **Extract Report Date:** Find the report date and standardize it to 'YYYY-MM-DD'.
        2. **Determine Report Type:** Identify the report type from the structure/content of the report (e.g., Liver Function Test, Kidney Function Test, Complete Blood Count, Lipid Profile). For this specific input, the report type is **Routine Urine Examination**.
        3. **Select Compulsory Parameters:** Extract **exactly four parameters** based on the report type. Since the input is a **Routine Urine Examination**, select the four most clinically relevant parameters from the available options. Use: `specificGravity`, `proteins`, `sugar`, and `pusCells`.
           (Ignore any other parameters in the report. If any of the four parameters are missing in the report, only include the ones present; do not add unrelated parameters.)
        4. **Create New JSON:** Return a JSON object with:
           - `reportDate`: standardized date
           - `healthParameters`: dictionary containing **only the four compulsory parameters**
        5. **Output Requirement:** Return only a single valid JSON object strictly following the Pydantic model format. Do not include explanations, extra text, or conversational notes.

        {format_instructions}
        '''


        prompt = PromptTemplate(
            input_variables=["context"],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm | parser
        
        # 2. Invoke the chain
        result = chain.invoke({"context": json_string})
        return result

    except Exception as e:
        print(f"Error during specific data extraction: {e}")
        return None

def get_health_score_dynamic_langchain(patient_data: Dict[str, Any], llm: ChatOpenAI) -> str:
    """
    Calculates a health score and provides a detailed clinical analysis.
    """
    try:
        # Extract Age and Gender from the nested structure
        age_str = patient_data.get("age") or patient_data.get("data", {}).get("age")
        gender = patient_data.get("gender") or patient_data.get("data", {}).get("gender") or patient_data.get("data", {}).get("sex")

        # Convert age to integer
        if isinstance(age_str, str):
            # Handles "23 Years" -> 23
            age = int(age_str.split()[0])
        elif isinstance(age_str, (int, float)):
            age = int(age_str)
        else:
            age = 30  # Default if missing, for LLM context

        # Ensure gender is a string, default to "Unknown"
        if not gender:
             gender = "Unknown"
        
        # Format the system and user prompts
        system_prompt_template = SystemMessagePromptTemplate.from_template(
            "You are an expert medical AI assistant with deep clinical knowledge. "
            "Your task is to analyze a user's medical lab report provided as JSON data "
            "along with the user's age and gender. Perform the following carefully:\n"
            "1. Interpret each lab parameter robustly, recognizing synonyms and variations in naming.\n"
            "2. For each lab parameter, use your internal knowledge of trusted clinical reference "
            "ranges to evaluate whether the value is normal, mildly abnormal, or "
            "significantly abnormal for the given age and gender.\n"
            "3. Start with a health score of 100. Deduct points based on the severity and "
            "clinical significance of each abnormal value, giving more weight to critical parameters.\n"
            "4. Output a final JSON with the following fields:\n"
            "   - \"score\" (number between 0 and 100, where 100 is perfect health)\n"
            "   - \"summary_reasoning\" (a brief, clear explanation of the overall score "
            "and main concerns)\n"
            "   - \"detailed_breakdown\" (an array of objects, one for each lab parameter analyzed. "
            "Each object should have:\n"
            "     - \"parameter\": The name of the lab test.\n"
            "     - \"value\": The patient's reported value.\n"
            "     - \"status\": \"Normal\", \"Mildly Abnormal\", or \"Significantly Abnormal\".\n"
            "     - \"analysis\": A concise explanation of why the value is concerning or normal, "
            "and its impact on the score.)\n"
            "5. The final output must be a single, valid JSON object, enclosed only by ```json ... ```.\n"
            "Always prioritize clinical validity, clarity, and patient safety in your reasoning."
        )

        user_message_template = HumanMessagePromptTemplate.from_template(
            "Patient Age: {age}\n"
            "Patient Gender: {gender}\n"
            "Lab Report Data:\n{lab_data}"
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_prompt_template, user_message_template]
        )

        formatted_prompt = chat_prompt.format_prompt(
            age=age,
            gender=gender,
            lab_data=json.dumps(patient_data, indent=2)
        )

        response = llm.invoke(formatted_prompt.to_messages())
        return response.content
    
    except Exception as e:
        return json.dumps({"score": -1, "reasoning": f"Error: {e}"})
