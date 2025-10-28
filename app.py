import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data
def load_data():
    movies_data = pd.read_csv('movies.csv')

    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = (
        movies_data['genres'] + " " +
        movies_data['keywords'] + " " +
        movies_data['tagline'] + " " +
        movies_data['cast'] + " " +
        movies_data['director']
    )

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    return movies_data, similarity

movies_data, similarity = load_data()

# UI
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites!")

movie_name = st.text_input("Enter a Movie Name 👇")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("⚠ Please type a movie name")
    else:
        all_titles = movies_data['title'].tolist()
        match = difflib.get_close_matches(movie_name, all_titles)

        if not match:
            st.error("❌ Movie not found! Try again.")
        else:
            close_match = match[0]
            st.write(f"*Searching recommendations for ➜ {close_match}*")

            movie_index = movies_data[movies_data.title == close_match].index[0]
            similarity_scores = list(enumerate(similarity[movie_index]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            st.subheader("🎯 Recommended Movies For You:")

            for i, (index, score) in enumerate(sorted_movies[1:15], start=1):
                movie = movies_data.iloc[index]

                st.markdown(f"""
                ### {i}. *{movie['title']}*
                ⭐ Rating: {movie['vote_average']}
                🎭 Genres: {movie['genres']}
                🎬 Director: {movie['director']}
                
                > {movie['overview'][:200]}...
                ---

                """)
