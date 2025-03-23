import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI setup
st.set_page_config(page_title="AI-Powered Smart Resume Screener and Ranker", layout="wide")
st.title("AI-Powered Smart Resume Screener and Ranker")

# File Upload
uploaded_files = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True, type=['pdf'])

# Job Description Input
job_description = st.text_area("Enter Job Description:")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Process Resumes
resume_data = []
if uploaded_files and job_description:
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
        resume_data.append({"Name": file.name, "Similarity Score": similarity_score, "Resume Text": resume_text})

# Convert to DataFrame
if resume_data:
    df = pd.DataFrame(resume_data).sort_values(by="Similarity Score", ascending=False)
    st.write("### Ranked Resumes")
    st.dataframe(df[['Name', 'Similarity Score']])

    # Visualization: Similarity Score Distribution
    st.write("### Similarity Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Similarity Score"], kde=True, bins=10, ax=ax)
    st.pyplot(fig)

    # Word Cloud for Skill Gap Analysis
    st.write("### Common Keywords in Resumes")
    all_text = " ".join(df["Resume Text"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Interactive Plotly Visualization
    st.write("### Candidate Score Analysis")
    fig = px.bar(df, x='Name', y='Similarity Score', color='Similarity Score', color_continuous_scale='Blues')
    st.plotly_chart(fig)

st.write("Upload PDFs and enter a job description to begin ranking candidates.")