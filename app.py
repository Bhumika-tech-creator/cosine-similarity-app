import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Cosine Similarity Calculator")

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

if st.button("Calculate Similarity"):
    if text1 and text2:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        st.success(f"Cosine Similarity: {similarity:.4f}")
    else:
        st.warning("Please enter both texts.")
