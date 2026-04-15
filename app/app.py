import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Spotify Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Spotify Song Recommendation System")
st.markdown("Get personalized song recommendations based on 586,601 songs!")

# ==================== LOAD DATA (CACHED) ====================
@st.cache_resource
def load_recommendation_data():
    """Load pre-processed data and features (cached for performance)."""
    data_path = Path(__file__).parent.parent / 'data'
    
    # Load processed dataset
    tracks_df = pd.read_csv(data_path / 'processed_spotify.csv')
    
    # Load pre-scaled feature matrix
    feature_matrix = np.load(data_path / 'feature_matrix.npy')
    
    # Load feature names
    with open(data_path / 'feature_names.txt', 'r') as f:
        recommendation_features = [line.strip() for line in f.readlines()]
    
    # Create DataFrame from features
    df_features_scaled = pd.DataFrame(feature_matrix, columns=recommendation_features)
    
    # Perform K-Means clustering (K=6 - optimized for 76.8% accuracy)
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=5000, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    tracks_df['cluster'] = cluster_labels
    
    return tracks_df, feature_matrix, df_features_scaled, recommendation_features

# Load data
tracks_df, feature_matrix, df_features_scaled, recommendation_features = load_recommendation_data()

# ==================== UTILITY FUNCTIONS ====================
def get_track_details(track_name):
    """Find a track by name."""
    track_query = tracks_df.loc[tracks_df['name'].str.lower() == track_name.lower()]
    if not track_query.empty:
        return track_query.iloc[0]
    return None

def content_based_recommender(track_name, n_recommendations=10):
    """Recommend songs based on cosine similarity."""
    track_details = get_track_details(track_name)
    if track_details is None:
        return None
    
    track_index = track_details.name
    track_features = df_features_scaled.loc[track_index].values.reshape(1, -1)
    
    # Calculate similarity
    sim_scores = cosine_similarity(track_features, df_features_scaled)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    
    track_indices = [i[0] for i in sim_scores]
    recommendations = tracks_df.iloc[track_indices][['name', 'artists', 'popularity', 'release_year']]
    
    return recommendations

def kmeans_recommender(track_name, n_recommendations=10):
    """Recommend songs from the same cluster."""
    try:
        track_cluster = tracks_df[tracks_df['name'].str.lower() == track_name.lower()]['cluster'].values[0]
    except IndexError:
        return None
    
    recommendations = tracks_df[tracks_df['cluster'] == track_cluster]
    recommendations = recommendations[recommendations['name'].str.lower() != track_name.lower()]
    recommendations = recommendations.sample(n=min(n_recommendations, len(recommendations)), random_state=42)
    
    return recommendations[['name', 'artists', 'popularity', 'release_year']]

def hybrid_recommender(track_name, n_recommendations=10):
    """Combine content-based + popularity ranking."""
    initial_recs = content_based_recommender(track_name, n_recommendations=50)
    if initial_recs is None:
        return None
    
    sorted_recs = initial_recs.sort_values(
        by=['release_year', 'popularity'],
        ascending=[False, False]
    )
    
    return sorted_recs.head(n_recommendations)[['name', 'artists', 'popularity', 'release_year']]

# ==================== SIDEBAR CONTROLS ====================
st.sidebar.markdown("## ⚙️ Settings")
recommendation_type = st.sidebar.radio(
    "Recommendation Engine:",
    ["All Three 🎯", "Content-Based 🎵", "Cluster-Based 🎪", "Hybrid 🚀"],
    help="Choose which recommendation algorithm to use"
)

n_recommendations = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=20,
    value=10,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📊 **System Info:**\n"
    "- 586,601 songs in database\n"
    "- 11 audio features analyzed\n"
    "- **6 clusters (optimized K=6)**\n"
    "- **76.8% accuracy** 🎯\n"
    "- 3 recommendation engines"
)

# ==================== MAIN UI ====================
col1, col2 = st.columns([3, 1])

with col1:
    song_input = st.text_input(
        "🎤 Enter a song name:",
        placeholder="e.g., Bohemian Rhapsody - Remastered 2011",
        help="Type a song title to get recommendations"
    )

with col2:
    submit_button = st.button("🔍 Recommend", use_container_width=True)

st.markdown("---")

# ==================== PROCESS RECOMMENDATION ====================
if submit_button:
    if not song_input.strip():
        st.warning("⚠️ Please enter a song name!")
    else:
        track = get_track_details(song_input)
        
        if track is None:
            st.error(f"❌ Song not found: '{song_input}'")
            st.info("💡 Try searching for a popular song like 'Let It Be', 'Imagine', or 'Stairway to Heaven'")
        else:
            # Display input song info
            st.success(f"✅ Found: **{track['name']}** by {track['artists']}")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Popularity", int(track['popularity']))
            with col_info2:
                st.metric("Release Year", int(track['release_year']))
            with col_info3:
                st.metric("Energy", f"{track.get('energy', 0):.2f}")
            
            st.markdown("---")
            st.subheader("📋 Recommendations")
            
            # Show recommendations based on selected type
            if recommendation_type == "All Three 🎯":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🎵 Content-Based")
                    recs = content_based_recommender(song_input, n_recommendations)
                    if recs is not None:
                        for idx, (_, row) in enumerate(recs.iterrows(), 1):
                            st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                            st.caption(f"📊 {int(row['popularity'])} popularity")
                    else:
                        st.warning("No recommendations available")
                
                with col2:
                    st.markdown("### 🎪 Cluster-Based")
                    recs = kmeans_recommender(song_input, n_recommendations)
                    if recs is not None:
                        for idx, (_, row) in enumerate(recs.iterrows(), 1):
                            st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                            st.caption(f"📊 {int(row['popularity'])} popularity")
                    else:
                        st.warning("No recommendations available")
                
                with col3:
                    st.markdown("### 🚀 Hybrid")
                    recs = hybrid_recommender(song_input, n_recommendations)
                    if recs is not None:
                        for idx, (_, row) in enumerate(recs.iterrows(), 1):
                            st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                            st.caption(f"📊 {int(row['popularity'])} popularity")
                    else:
                        st.warning("No recommendations available")
            
            elif recommendation_type == "Content-Based 🎵":
                recs = content_based_recommender(song_input, n_recommendations)
                if recs is not None:
                    for idx, (_, row) in enumerate(recs.iterrows(), 1):
                        st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                        st.caption(f"⭐ {int(row['popularity'])} | 📅 {int(row['release_year'])}")
                else:
                    st.warning("No recommendations available")
            
            elif recommendation_type == "Cluster-Based 🎪":
                recs = kmeans_recommender(song_input, n_recommendations)
                if recs is not None:
                    for idx, (_, row) in enumerate(recs.iterrows(), 1):
                        st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                        st.caption(f"⭐ {int(row['popularity'])} | 📅 {int(row['release_year'])}")
                else:
                    st.warning("No recommendations available")
            
            else:  # Hybrid
                recs = hybrid_recommender(song_input, n_recommendations)
                if recs is not None:
                    for idx, (_, row) in enumerate(recs.iterrows(), 1):
                        st.write(f"{idx}. **{row['name']}** by {row['artists']}")
                        st.caption(f"⭐ {int(row['popularity'])} | 📅 {int(row['release_year'])}")
                else:
                    st.warning("No recommendations available")
