import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("jobs.csv")

# Preprocessing
df = df[['job_title', 'job_description', 'sector', 'location']]
df = df.dropna()
df['combined'] = df['job_title'] + " " + df['job_description'] + " " + df['sector'] + " " + df['location']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Recommendation Function
def recommend_jobs(user_input):
    user_vec = tfidf.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)
    top_jobs = sim_scores[0].argsort()[-5:][::-1]
    return df.iloc[top_jobs][['job_title', 'location', 'sector']]

# UI
st.title("Job Recommendation System")

user_input = st.text_input("Enter your skills (Example: Python, ML, Data Analysis)")

if st.button("Recommend"):
    results = recommend_jobs(user_input)
    st.write(results)