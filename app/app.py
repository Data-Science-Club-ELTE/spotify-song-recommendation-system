import streamlit as st

st.set_page_config(page_title="Spotify Recommender", layout="centered")
st.title("Spotify Song Recommendation System")
st.write("Milestone 1: simple UI + deployable app.")

song = st.text_input("Enter a song name")
if st.button("Recommend"):
    if not song.strip():
        st.warning("Please type a song name first.")
    else:
        st.success(f"Placeholder recommendations for: {song}")
        st.write(["Recommendation 1", "Recommendation 2", "Recommendation 3"])
