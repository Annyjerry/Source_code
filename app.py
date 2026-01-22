import os
import sys
# Manually add the torch lib path to the environment
torch_lib_path = r"C:\Users\Anny Jerry\anaconda3\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Set page to wide mode to fit two sections on one screen
st.set_page_config(page_title="Code Analyzer API", layout="wide")

# --- Load Components ---
@st.cache_resource
def load_assets():
    model = joblib.load('api_assets/Neural.joblib')
    vectorizer = joblib.load('api_assets/tfidf_vectorizer.joblib')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
    return model, vectorizer, tokenizer, bert_model

nn_model, tfidf_vec, tokenizer, bert_model = load_assets()

# --- Helper: Feature Extraction ---
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def is_valid_code(text):
    # Basic check: Code usually contains specific symbols or keywords
    code_indicators = ['def ', 'import ', '{', '}', ';', 'func ', 'var ', 'public ', 'class ']
    return any(indicator in text for indicator in code_indicators) or len(text.split()) > 3

# --- UI Layout ---
st.title("🛡️ Source Code Severity Analyzer")
st.markdown("---")

# Create two sections (Columns)
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("1. User Input")
    source_code = st.text_area("Enter Source Code here:", height=300, placeholder="Paste your code snippet...")
    language = st.selectbox("Select Programming Language:", ["Python", "Java", "C++", "JavaScript", "Go"])
    
    analyze_btn = st.button("Analyze Code", type="primary")

with right_col:
    st.header("2. Result Info")
    
    if analyze_btn:
        if not source_code.strip() or not is_valid_code(source_code):
            st.error("⚠️ This is not a code to analyze. Please provide valid source code.")
        else:
            # 1. Extract BERT features
            bert_feat = get_embeddings(source_code).reshape(1, -1)
            
            # 2. Extract TF-IDF features
            tfidf_feat = tfidf_vec.transform([source_code]).toarray()
            
            # 3. Combine Features
            combined_feat = np.hstack([bert_feat, tfidf_feat])
            
            # 4. Predict
            prediction = nn_model.predict(combined_feat)[0]
            probs = nn_model.predict_proba(combined_feat)[0]
            
            st.success("Analysis Complete!")
            
            # Hidden until clicked using expander
            with st.expander("Click to view Analysis Results"):
                st.write(f"**Predicted Severity:** {prediction}")
                st.write("**Confidence Scores:**")
                
                # Show probability bar chart
                prob_df = pd.DataFrame({
                    "Level": nn_model.classes_,
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Level"))