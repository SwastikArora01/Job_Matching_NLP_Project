import streamlit as st
import pandas as pd
import PyPDF2
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

# ------------------------------
# Load DistilBERT once
# ------------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------
# Extract text from PDF
# ------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ------------------------------
# Generate embeddings
# ------------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# ------------------------------
# Streamlit GUI
# ------------------------------
st.title("📄 Resume - Job Description Matcher")

# Upload job description
st.subheader("Upload Job Description (PDF)")
job_file = st.file_uploader("Choose a job description PDF", type=["pdf"])

# Upload resumes
st.subheader("Upload Candidate Resumes (PDFs)")
resume_files = st.file_uploader("Choose one or more resumes", type=["pdf"], accept_multiple_files=True)

if job_file and resume_files:
    # Extract job description text
    job_text = extract_text_from_pdf(job_file)
    job_embedding = get_embedding(job_text)

    results = []

    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        resume_embedding = get_embedding(resume_text)

        # Compute cosine similarity
        similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]

        results.append({
            "Resume File": resume_file.name,
            "Match Score (%)": round(similarity * 100, 2)
        })

    # Create DataFrame
    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)

    st.subheader("📊 Matching Results")
    st.dataframe(df, use_container_width=True)

    # Highlight best match
    best_match = df.iloc[0]
    st.success(f"✅ Best Match: **{best_match['Resume File']}** with score {best_match['Match Score (%)']}%")
