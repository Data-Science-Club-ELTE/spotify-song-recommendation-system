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
    page_title="Spotify Recommender (Demo)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Spotify Song Recommendation System - Demo Version")
st.markdown("Get personalized song recommendations from a curated 50K song demo database!")
st.info("💡 **Demo Version**: Using 50,000 songs for instant cloud deployment. Full version (586K songs) available locally.")

# ==================== LOAD DEMO DATA ====================
@st.cache_resource
def load_demo_data():
    """Load or generate demo data (50k sample)."""
    data_path = Path(__file__).parent.parent / 'data'
    demo_path = Path(__file__).parent / 'demo_data'
    demo_path.mkdir(exist_ok=True)
    
    # Check if demo files exist
    demo_csv = demo_path / 'demo_spotify.csv'
    demo_features = demo_path / 'demo_features.npy'
    demo_names = demo_path / 'demo_feature_names.txt'
    
    if not demo_csv.exists():
        st.info("� Generating demo dataset...")
        
        # Try to load full dataset from local
        if (data_path / 'processed_spotify.csv').exists():
            st.info("✅ Found local data files!")
            tracks_df = pd.read_csv(data_path / 'processed_spotify.csv')
            feature_matrix = np.load(data_path / 'feature_matrix.npy')
            with open(data_path / 'feature_names.txt', 'r') as f:
                recommendation_features = [line.strip() for line in f.readlines()]
        else:
            # Use synthetic data for fast demo
            st.info("🎯 Using synthetic demo data (upload real data to GitHub Releases to use your own dataset)")
            n_songs = 50000
            features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
            
            data = {
                'name': [f'Song {i}' for i in range(n_songs)],
                'artists': [f'Artist {i % 100}' for i in range(n_songs)],
                'popularity': np.random.randint(0, 100, n_songs),
                'release_year': np.random.randint(1950, 2024, n_songs)
            }
            tracks_df = pd.DataFrame(data)
            feature_matrix = np.random.rand(n_songs, 11)
            recommendation_features = features
        
        # Use 50k sample
        sample_size = min(50000, len(tracks_df))
        sample_indices = np.random.choice(len(tracks_df), size=sample_size, replace=False)
        tracks_df = tracks_df.iloc[sample_indices].reset_index(drop=True)
        feature_matrix = feature_matrix[sample_indices]
        
        # Save demo data
        tracks_df.to_csv(demo_csv, index=False)
        np.save(demo_features, feature_matrix)
        with open(demo_names, 'w') as f:
            f.write('\n'.join(recommendation_features))
        
        st.success(f"✅ Demo dataset ready! ({len(tracks_df):,} songs)")
    else:
        tracks_df = pd.read_csv(demo_csv)
        feature_matrix = np.load(demo_features)
        with open(demo_names, 'r') as f:
            recommendation_features = [line.strip() for line in f.readlines()]
    
    # Create DataFrame from features
    df_features_scaled = pd.DataFrame(feature_matrix, columns=recommendation_features)
    
    # Perform K-Means clustering (K=6 - optimized)
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=100, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    tracks_df['cluster'] = cluster_labels
    
    return tracks_df, feature_matrix, df_features_scaled, recommendation_features

# Load demo data
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
    return sorted_recs.head(n_recommendations)

# ==================== SIDEBAR ====================
st.sidebar.title("📊 System Info")
st.sidebar.write(f"**Total Songs:** {len(tracks_df):,}")
st.sidebar.write(f"**Features:** {len(recommendation_features)}")
st.sidebar.write(f"**Clusters:** 6")
st.sidebar.write(f"**Accuracy:** ~76.8%")

# ==================== MAIN APP ====================
st.header("🔍 Find Similar Songs")

# Song search
track_name = st.text_input("Enter a song name:", placeholder="e.g., Bohemian Rhapsody")

if track_name:
    track_details = get_track_details(track_name)
    
    if track_details is None:
        st.warning(f"❌ Song '{track_name}' not found. Try another song!")
        st.info("💡 Sample songs: " + ", ".join(tracks_df['name'].head(5).tolist()))
    else:
        st.success(f"✅ Found: **{track_details['name']}** by {track_details['artists']}")
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
        
        # Choose recommendation method
        tab1, tab2, tab3 = st.tabs(["📍 Content-Based", "🎯 Cluster-Based", "🔀 Hybrid"])
        
        with tab1:
            st.subheader("Content-Based (Audio Features)")
            recs = content_based_recommender(track_name, n_recs)
            if recs is not None:
                st.dataframe(recs, use_container_width=True)
            else:
                st.error("Could not generate recommendations")
        
        with tab2:
            st.subheader("Cluster-Based (Similar Vibe)")
            recs = kmeans_recommender(track_name, n_recs)
            if recs is not None and len(recs) > 0:
                st.dataframe(recs, use_container_width=True)
            else:
                st.warning("No recommendations in this cluster")
        
        with tab3:
            st.subheader("Hybrid (Content + Popularity)")
            recs = hybrid_recommender(track_name, n_recs)
            if recs is not None:
                st.dataframe(recs, use_container_width=True)
            else:
                st.error("Could not generate recommendations")

# ==================== STATS SECTION ====================
st.divider()
st.header("📈 Dataset Statistics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Songs", f"{len(tracks_df):,}")
with col2:
    st.metric("Popularity Range", f"{int(tracks_df['popularity'].min())}-{int(tracks_df['popularity'].max())}")
with col3:
    st.metric("Year Range", f"{int(tracks_df['release_year'].min())}-{int(tracks_df['release_year'].max())}")

st.subheader("🎵 Popular Songs")
st.dataframe(tracks_df.nlargest(10, 'popularity')[['name', 'artists', 'popularity', 'release_year']], use_container_width=True)

st.divider()
st.info("🚀 **Want full 586K song database?** Run locally: `streamlit run app/app.py`")
