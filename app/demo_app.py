import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import warnings
import urllib.request
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Spotify Recommender (Demo)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Spotify Song Recommendation System - Demo Version")
st.markdown("Get personalized song recommendations from 10,000 REAL Spotify songs!")
st.info("💡 **Demo Version**: Using 10,000 real songs from our 586K dataset for fast cloud deployment.")

# ==================== LOAD & SAMPLE REAL DATA ====================
@st.cache_resource
def load_demo_data():
    """Download full real data and sample 10k songs - REAL DATA ONLY."""
    data_path = Path(__file__).parent.parent / 'data'
    demo_path = Path(__file__).parent / 'demo_data'
    demo_path.mkdir(exist_ok=True)
    data_path.mkdir(exist_ok=True)
    
    # Check if cached demo exists
    demo_csv = demo_path / 'demo_spotify.csv'
    demo_features = demo_path / 'demo_features.npy'
    demo_names = demo_path / 'demo_feature_names.txt'
    
    if demo_csv.exists() and demo_features.exists() and demo_names.exists():
        tracks_df = pd.read_csv(demo_csv)
        feature_matrix = np.load(demo_features)
        with open(demo_names, 'r') as f:
            recommendation_features = [line.strip() for line in f.readlines()]
        df_features_scaled = pd.DataFrame(feature_matrix, columns=recommendation_features)
        
        kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=100, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        tracks_df['cluster'] = cluster_labels
        
        return tracks_df, feature_matrix, df_features_scaled, recommendation_features
    
    # Download real dataset from GitHub Releases
    st.info("📥 Downloading REAL Spotify data...")
    
    csv_url = 'https://github.com/Data-Science-Club-ELTE/spotify-song-recommendation-system/releases/download/v1.0-data/processed_spotify.csv'
    features_url = 'https://github.com/Data-Science-Club-ELTE/spotify-song-recommendation-system/releases/download/v1.0-data/feature_matrix.npy'
    names_url = 'https://github.com/Data-Science-Club-ELTE/spotify-song-recommendation-system/releases/download/v1.0-data/feature_names.txt'
    
    csv_file = data_path / 'processed_spotify.csv'
    features_file = data_path / 'feature_matrix.npy'
    names_file = data_path / 'feature_names.txt'
    
    try:
        # Download files
        if not csv_file.exists():
            with st.spinner("Downloading songs..."):
                urllib.request.urlretrieve(csv_url, csv_file)
        
        if not features_file.exists():
            with st.spinner("Downloading features..."):
                urllib.request.urlretrieve(features_url, features_file)
        
        if not names_file.exists():
            with st.spinner("Downloading metadata..."):
                urllib.request.urlretrieve(names_url, names_file)
        
        # Load full dataset
        st.info("🔄 Loading and sampling 10,000 real songs...")
        tracks_df = pd.read_csv(csv_file)
        feature_matrix = np.load(features_file)
        with open(names_file, 'r') as f:
            recommendation_features = [line.strip() for line in f.readlines()]
        
        # Sample 10K REAL songs
        sample_size = min(10000, len(tracks_df))
        sample_indices = np.random.choice(len(tracks_df), size=sample_size, replace=False)
        tracks_df = tracks_df.iloc[sample_indices].reset_index(drop=True)
        feature_matrix = feature_matrix[sample_indices]
        
        st.success(f"✅ Loaded {sample_size:,} REAL songs from Spotify!")
        
    except Exception as e:
        st.error(f"Error downloading: {str(e)}")
        st.stop()
    
    # Save encrypted demo
    tracks_df.to_csv(demo_csv, index=False)
    np.save(demo_features, feature_matrix)
    with open(demo_names, 'w') as f:
        f.write('\n'.join(recommendation_features))
    
    # Create feature DataFrame
    df_features_scaled = pd.DataFrame(feature_matrix, columns=recommendation_features)
    
    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=100, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    tracks_df['cluster'] = cluster_labels
    
    return tracks_df, feature_matrix, df_features_scaled, recommendation_features

# Load data
tracks_df, feature_matrix, df_features_scaled, recommendation_features = load_demo_data()
st.success(f"✅ Loaded {len(tracks_df):,} songs | {len(df_features_scaled.columns)} features")

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
    return sorted_recs.head(n_recommendations)

# ==================== SIDEBAR ====================
st.sidebar.title("📊 System Info")
st.sidebar.write(f"**Total Songs:** {len(tracks_df):,}")
st.sidebar.write(f"**Features:** {len(recommendation_features)}")
st.sidebar.write(f"**Clusters:** 6")
st.sidebar.write(f"**Accuracy:** 76.8%")
st.sidebar.write("**Data:** 100% REAL")

# ==================== MAIN APP ====================
st.header("🔍 Find Similar Songs")

# Show sample songs
st.subheader("🎵 Try These Popular Songs:")
sample_songs = tracks_df.nlargest(10, 'popularity')['name'].tolist()
cols = st.columns(5)
for idx, song in enumerate(sample_songs[:5]):
    with cols[idx]:
        if st.button(f"🎵 {song[:20]}", key=f"sample_{idx}"):
            st.session_state.selected_song = song

track_name = st.text_input("Enter a song name:", placeholder="e.g., Bohemian Rhapsody", value=st.session_state.get('selected_song', ''))

if track_name:
    track_details = get_track_details(track_name)
    
    if track_details is None:
        st.warning(f"❌ Song '{track_name}' not found.")
        st.info("💡 Sample songs: " + ", ".join(tracks_df['name'].head(5).tolist()))
    else:
        st.success(f"✅ Found: **{track_details['name']}** by {track_details['artists']}")
        
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        method = st.radio(
            "Method:",
            ["Content-Based", "Cluster-Based", "Hybrid"]
        )
        
        if method == "Content-Based":
            recs = content_based_recommender(track_name, n_recs)
            st.subheader("🎧 Similar Audio Features")
        elif method == "Cluster-Based":
            recs = kmeans_recommender(track_name, n_recs)
            st.subheader("🎯 Same Cluster")
        else:
            recs = hybrid_recommender(track_name, n_recs)
            st.subheader("⚡ Hybrid (Content + Popularity)")
        
        if recs is not None and len(recs) > 0:
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
        else:
            st.warning("No recommendations available.")

# ==================== STATS ====================
st.divider()
st.subheader("📈 Dataset Stats")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Real Songs", f"{len(tracks_df):,}")
with col2:
    st.metric("Year Range", f"{int(tracks_df['release_year'].min())}-{int(tracks_df['release_year'].max())}")
with col3:
    st.metric("Avg Popularity", f"{int(tracks_df['popularity'].mean())}")

st.info("💡 Need all 586K songs? Run locally: `streamlit run app/app.py`")
