import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Cosine Similarity Calculator - Advanced")

option = st.radio("Choose input method:", ["Compare 2 Texts", "Multiple Lines"])

if option == "Compare 2 Texts":
    text1 = st.text_area("Enter first text:")
    text2 = st.text_area("Enter second text:")

    if st.button("Calculate Similarity"):
        if text1 and text2:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([text1, text2])
            score = cosine_similarity(vectors[0], vectors[1])[0][0]
            st.success(f"Similarity: {score:.4f}")
        else:
            st.warning("Please fill both texts.")

elif option == "Multiple Lines":
    multiline = st.text_area("Enter one sentence per line:")
    if st.button("Compare All"):
        lines = [line.strip() for line in multiline.split('\n') if line.strip()]
        if len(lines) >= 2:
            tfidf = TfidfVectorizer().fit_transform(lines)
            similarity_matrix = cosine_similarity(tfidf)
            st.dataframe(similarity_matrix)
        else:
            st.warning("Enter at least 2 sentences.")
