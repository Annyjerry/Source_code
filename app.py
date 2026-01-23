import os
import sys
import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# 1. FIX: Handle DLL initialization for Torch on Windows
try:
    torch_lib_path = r"C:\Users\Anny Jerry\anaconda3\Lib\site-packages\torch\lib"
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
except Exception:
    pass

# Set page to wide mode to fit two sections on one screen
st.set_page_config(page_title="Code Analyzer API", layout="wide")

# --- 2. Knowledge Base Definition ---
# Ensure the keys (Low, Medium, etc.) match exactly what your model outputs
SEVERITY_KNOWLEDGE_BASE = {
    0: {
        "description": "The code contains minor stylistic issues or non-optimal patterns that do not immediately threaten security.",
        "recommendation": "Review naming conventions, remove dead code, and ensure comments are up to date. No urgent security patch required."
    },
    1: {
        "description": "Potential logical vulnerabilities detected. These could lead to unexpected behavior or information leakage under specific edge cases.",
        "recommendation": "Improve error handling and ensure all input variables are typed correctly. Add unit tests for boundary conditions."
    },
    2: {
        "description": "Significant security risks identified. Patterns found are common in memory leaks, unauthorized access, or logic flaws.",
        "recommendation": "Implement strict input sanitization. Avoid using deprecated functions. Conduct a manual code review of the affected logic."
    },
    3: {
        "description": "Dangerous patterns detected (e.g., potential SQL Injection, Buffer Overflow, or Hardcoded Credentials).",
        "recommendation": "IMMEDIATE ACTION: Use parameterized queries, implement robust authentication, and use secure memory-safe libraries."
    }
}

# --- 3. Load Components ---
@st.cache_resource
def load_assets():
    # Paths based on your previous messages
    model = joblib.load('api_assets/Neural.joblib')
    vectorizer = joblib.load('api_assets/tfidf_vectorizer.joblib')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
    return model, vectorizer, tokenizer, bert_model

nn_model, tfidf_vec, tokenizer, bert_model = load_assets()

# --- 4. Helper: Feature Extraction ---
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def is_valid_code(text):
    code_indicators = ['def ', 'import ', '{', '}', ';', 'func ', 'var ', 'public ', 'class ', 'print(']
    return any(indicator in text for indicator in code_indicators) or len(text.split()) > 3

# --- 5. UI Layout ---
st.title("🛡️ Source Code Vulnerability Severity Analyzer")
st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("1. User Input")
    source_code = st.text_area("Enter Source Code here:", height=300, placeholder="Paste your code snippet...")
    language = st.selectbox("Select Programming Language:", ["Python", "Java", "C++", "JavaScript", "C", "php"])
    
    analyze_btn = st.button("Analyze Code", type="primary")

with right_col:
    st.header("2. Result Info")
    
    if analyze_btn:
        if not source_code.strip() or not is_valid_code(source_code):
            st.error("⚠️ This is not a code to analyze. Please provide valid source code.")
        else:
            with st.spinner('Model is analyzing patterns...'):
                # 1. Feature Extraction
                bert_feat = get_embeddings(source_code).reshape(1, -1)
                tfidf_feat = tfidf_vec.transform([source_code]).toarray()
                combined_feat = np.hstack([bert_feat, tfidf_feat])
                
                # 2. Prediction
                prediction = nn_model.predict(combined_feat)[0]
                probs = nn_model.predict_proba(combined_feat)[0]
                conf_score = max(probs) * 100
            
            st.success("Analysis Complete!")
            
            # 3. Results Section (Hidden until expander clicked)
            with st.expander("Click to view Analysis Results", expanded=True):
                st.subheader(f"Predicted Severity: {prediction}")
                
                # Knowledge Base Integration
                info = SEVERITY_KNOWLEDGE_BASE.get(prediction, {"description": "N/A", "recommendation": "N/A"})
                
                st.info(f"**Reasoning:** The system detected structural patterns in your **{language}** code "
                        f"consistent with **{prediction}** risk levels with **{conf_score:.2f}%** confidence.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📖 Description")
                    st.write(info["description"])
                with col2:
                    st.markdown("### 🛠️ Recommended Fixes")
                    st.write(info["recommendation"])
                
                st.markdown("---")
                st.write("**Full Confidence Distribution:**")
                prob_df = pd.DataFrame({
                    "Level": nn_model.classes_,
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Level"))