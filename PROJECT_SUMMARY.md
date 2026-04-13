# 🎵 Spotify Song Recommendation System — Complete Project Summary

> This document is the single source of truth for everything that has been planned, built, and updated in this project, from the very first commit to the latest changes.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team](#2-team)
3. [Problem Statement](#3-problem-statement)
4. [Challenges](#4-challenges)
5. [Tools & Technologies](#5-tools--technologies)
6. [Repository Structure](#6-repository-structure)
7. [Phase 1 — Repository Setup & Scaffold](#7-phase-1--repository-setup--scaffold)
8. [Phase 2 — Data Exploration & Cleaning](#8-phase-2--data-exploration--cleaning)
9. [Phase 3 — Machine Learning Recommendation Models](#9-phase-3--machine-learning-recommendation-models)
10. [Phase 4 — Web Application](#10-phase-4--web-application)
11. [Phase 5 — Documentation Updates](#11-phase-5--documentation-updates)
12. [Current Status](#12-current-status)
13. [Suggested Next Steps](#13-suggested-next-steps)

---

## 1. Project Overview

The **Spotify Song Recommendation System** is an end-to-end data science project that suggests songs similar to a user's input based on audio characteristics such as tempo, energy, danceability, and more. It uses machine learning techniques to identify similarity between songs and exposes recommendations through an interactive Streamlit web interface.

---

## 2. Team

| Name               | Role / Responsibilities                                         |
| :----------------- | :-------------------------------------------------------------- |
| Ayush Khemani      | Project Co-Lead, Project Coordination, GitHub Management        |
| Haseeb Raza        | Project Co-Lead, Technical Support, Team Coordination           |
| Nursultan Tuleev   | Build ML Recommendation Model                                   |
| Muna Hassan        | Data Exploration, Cleaning, EDA                                 |
| Saidul Islam Nayan | Build ML Recommendation Model                                   |
| Yushay Aizaz       | Create Web Interface using Streamlit                            |

---

## 3. Problem Statement

With millions of songs available across streaming platforms, users often struggle to discover music that matches their preferences or mood. Traditional browsing relies heavily on manual searching or popularity-based suggestions, which miss personalised taste.

This project solves that problem by building a **content-based recommendation system** that:

- Analyses the audio features of a song the user already knows.
- Identifies songs with the most similar audio profiles.
- Returns ranked recommendations automatically, without needing any user rating history.

---

## 4. Challenges

| Challenge | Details |
| :-- | :-- |
| Large dataset | 600k+ songs — memory and compute efficiency matter |
| Messy real-world data | Missing values, incorrect types, string-encoded lists |
| Feature selection | Choosing which audio features best capture musical similarity |
| Algorithm design | Comparing content-based, cluster-based, and hybrid strategies |
| App integration | Connecting the ML models to the Streamlit front-end |
| Speed | Keeping recommendation latency acceptable for an interactive UI |

---

## 5. Tools & Technologies

| Category | Tools |
| :-- | :-- |
| Language | Python 3 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn (`StandardScaler`, `KMeans`, `MiniBatchKMeans`, `cosine_similarity`, `TSNE`, `silhouette_score`, `davies_bouldin_score`) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Web interface | Streamlit |
| Version control | Git & GitHub |

---

## 6. Repository Structure

```
spotify-song-recommendation-system/
├── app/
│   └── app.py                     # Streamlit web application
├── data/
│   ├── raw/
│   │   └── tracks.csv             # Original 600k+ song dataset (not committed)
│   └── processed/                 # Placeholder for processed outputs
├── models/
│   └── recommendations.ipynb      # ML recommendation models notebook
├── notebooks/
│   └── dataset_exploration.ipynb  # EDA & data cleaning notebook
├── src/                           # Source code placeholder
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── PROJECT_SUMMARY.md             # ← This file
```

---

## 7. Phase 1 — Repository Setup & Scaffold

### What was done

- Initialised the GitHub repository with a standard Python `.gitignore`.
- Created the folder skeleton: `app/`, `data/raw/`, `data/processed/`, `models/`, `notebooks/`, `src/` (each with a `.gitkeep` so empty folders are tracked by Git).
- Added `requirements.txt` with the initial dependency (`streamlit`).
- Wrote the initial `README.md` with project description, team table, problem statement, challenges, expectations, tools list, and how-to-run instructions.
- Set up branch protection and a pull-request workflow for collaborative development.

### Key files introduced

| File | Purpose |
| :-- | :-- |
| `README.md` | Project overview and instructions |
| `requirements.txt` | Dependency list |
| `.gitignore` | Excludes raw data archives, `__pycache__`, IDE files, OS files |

---

## 8. Phase 2 — Data Exploration & Cleaning

**Notebook:** `notebooks/dataset_exploration.ipynb`

### Dataset

- **Source:** Spotify Tracks dataset (CSV), placed at `data/raw/tracks.csv`.
- **Size:** ~600,000 rows, 20 columns.
- **Key columns:** `id`, `name`, `artists`, `id_artists`, `release_date`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `explicit`, `popularity`.

### Steps performed

#### 8.1 Loading & Inspection
```python
tracks_df = pd.read_csv('../data/raw/tracks.csv')
tracks_df.info()          # column names, dtypes, null counts
tracks_df.head()          # first 5 rows
```

#### 8.2 Missing Value Analysis
- Counted missing values per column and expressed as a percentage of total rows.
- Visualised missing counts as a horizontal bar chart (Matplotlib/Seaborn).
- **Decision:** Dropped any row where the song `name` is null (a song without a name cannot be usefully recommended).

#### 8.3 Duplicate Removal
- Detected fully duplicate rows with `df.duplicated()`.
- Also checked for duplicate values in the first column (song ID).
- Dropped all duplicate rows with `drop_duplicates()`.
- Printed before/after row counts.

#### 8.4 Data-Type Fixes
| Column | Original Type | Fixed Type | Method |
| :-- | :-- | :-- | :-- |
| `artists` | `str` (Python list literal) | `list` | `ast.literal_eval` with fallback to `["Unknown Artist"]` |
| `id_artists` | `str` (Python list literal) | `list` | `ast.literal_eval` with fallback to `["unknown_artist_id"]` |
| `explicit` | `int` (0/1) | `bool` | `.astype(bool)` |
| `mode` | `int` (0/1) | `bool` | `.astype(bool)` |
| `release_date` | `str` (YYYY-MM-DD or YYYY) | `int` (year only) | `str[:4]` → `pd.to_numeric` → drop un-parseable |

The original `release_date` column was dropped after `release_year` was created.

#### 8.5 Column Name Normalisation
```python
tracks_cleaned.columns = (
    tracks_cleaned.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_', regex=False)
    .str.replace('-', '_', regex=False)
)
```

#### 8.6 Saving the Cleaned Dataset
```python
tracks_cleaned.to_csv('../data/cleaned_spotify.csv', index=False)
```

### Output
- **File:** `data/cleaned_spotify.csv`
- Standardised column names, correct data types, no missing names, no duplicate rows.

---

## 9. Phase 3 — Machine Learning Recommendation Models

**Notebook:** `models/recommendations.ipynb`

### Audio Feature Set

Eleven features were selected for all models:

| Feature | Description |
| :-- | :-- |
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Intensity and activity level (0–1) |
| `key` | Musical key (0–11, integer) |
| `loudness` | Overall loudness in dB |
| `mode` | Major (1) or minor (0) |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Likelihood of no vocals (0–1) |
| `liveness` | Presence of live audience (0–1) |
| `valence` | Musical positivity (0–1) |
| `tempo` | Estimated beats per minute |

### Preprocessing
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_features_scaled = scaler.fit_transform(df_features)
```
All 11 features are standardised (zero mean, unit variance) so that features on different scales don't dominate the similarity calculation.

---

### Model A — Content-Based Recommender (Cosine Similarity)

**How it works:**
1. Given an input song name, find its row in the dataset.
2. Retrieve its scaled feature vector.
3. Compute pairwise cosine similarity between that vector and every other track.
4. Sort by descending similarity and return the top-N results (excluding the input track itself).

```python
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommender(track_name, n_recommendations=10):
    track_features = df_features_scaled.loc[track_index].values.reshape(1, -1)
    sim_scores = cosine_similarity(track_features, df_features_scaled)
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    return tracks_df.iloc[top_indices]
```

**Strengths:** Deterministic, interpretable, works well when audio features are distinctive.  
**Limitation:** Computationally expensive at O(N) per query on 600k songs.

---

### Model B — K-Means Clustering Recommender

**How it works:**
1. Run the **Elbow Method** — plot SSE vs. K from 2 to 14 to find the point of diminishing returns.
2. Train `MiniBatchKMeans` with the chosen K=7 on the full scaled feature matrix.
3. Assign every song to a cluster.
4. For a query song, find its cluster label, then randomly sample N songs from the same cluster.

```python
from sklearn.cluster import MiniBatchKMeans

K = 7
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048, n_init='auto')
tracks_df['cluster_kmeans'] = kmeans.fit_predict(df_features_scaled)
```

**Strengths:** Very fast at inference time (just a cluster lookup + sample).  
**Limitation:** Songs within a cluster may still vary widely; randomness means different runs return different results.

---

### Model C — Hybrid Recommender (Content + Popularity)

**How it works:**
1. Use the content-based recommender to fetch a larger pool of 50 candidate songs.
2. Re-rank the candidates by `release_year` (descending) then `popularity` (descending).
3. Return the top-N from this re-ranked list.

```python
def hybrid_recommender(track_name, n_recommendations=10):
    pool = content_based_recommender(track_name, n_recommendations=50, display=False)
    pool['year'] = pd.to_datetime(pool['release_date'], format='mixed').dt.year
    return pool.sort_values(['year','popularity'], ascending=False).head(n_recommendations)
```

**Strengths:** Surfaces recent, popular songs that are still audio-similar to the input.  
**Limitation:** May bias toward mainstream songs even when the user prefers niche music.

---

### Visualisation & Evaluation

#### t-SNE Cluster Visualisation
- Reduces the 11-dimensional feature space to 2D using t-SNE.
- Scatter plot coloured by K-Means cluster label to visually inspect cluster separation.

#### Radar Chart (Audio Feature Comparison)
- Uses Plotly `Scatterpolar` to overlay two tracks' audio profiles on a radar/spider chart.
- Lets users intuitively see why two songs are considered similar.

#### Clustering Evaluation Metrics
| Metric | Interpretation |
| :-- | :-- |
| **Silhouette Score** | Ranges −1 to 1; higher = better-defined clusters |
| **Davies-Bouldin Index** | Lower = better; measures average similarity between clusters |

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
silhouette  = silhouette_score(df_features_scaled, tracks_df['cluster_kmeans'])
db_index    = davies_bouldin_score(df_features_scaled, tracks_df['cluster_kmeans'])
```

---

## 10. Phase 4 — Web Application

**File:** `app/app.py`

A minimal **Streamlit** application that provides an interactive UI for the recommendation system.

### Current implementation (Milestone 1 scaffold)
```python
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
```

### How to run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app
streamlit run app/app.py
```

The app opens at `http://localhost:8501` in your browser.

### What the app does right now
- Accepts a song name in a text input field.
- Shows a warning if the field is empty.
- Returns hard-coded placeholder recommendations (ML integration is the next step).

---

## 11. Phase 5 — Documentation Updates

The `README.md` was updated to include a **"What Has Been Built"** section documenting:
- The data pipeline and EDA notebook.
- All three recommendation models.
- The Streamlit web app.
- A current-status checklist.

This `PROJECT_SUMMARY.md` was created as a standalone, comprehensive reference document.

---

## 12. Current Status

| Area | Status |
| :-- | :-- |
| Repository scaffold & folder structure | ✅ Complete |
| Raw dataset (`data/raw/tracks.csv`) | ✅ Available (not committed — too large) |
| EDA & data cleaning notebook | ✅ Complete |
| Cleaned dataset output | ✅ Produced by running EDA notebook |
| Content-based recommender (cosine similarity) | ✅ Complete |
| K-Means clustering recommender | ✅ Complete |
| Hybrid recommender (content + popularity) | ✅ Complete |
| Visualisations (t-SNE, radar chart) | ✅ Complete |
| Clustering evaluation metrics | ✅ Complete |
| Streamlit web app (scaffold) | ✅ Runnable |
| ML models wired into Streamlit app | ⏳ In progress |
| Model saved to disk (`.pkl` or similar) | ⏳ Not yet done |
| Deployed app (e.g., Streamlit Cloud) | ⏳ Not yet done |

---

## 13. Suggested Next Steps

The following improvements would bring the project to a fully polished state:

### High Priority

1. **Integrate ML models into `app/app.py`**
   - Load `cleaned_spotify.csv` at app startup.
   - Run `StandardScaler` + cosine similarity (or load a pre-fitted scaler) when the user submits a song.
   - Display real recommendations with song name, artist, and a similarity score.

2. **Save and load the fitted scaler and cluster labels**
   - Use `joblib` or `pickle` to serialise the `StandardScaler` and `MiniBatchKMeans` model.
   - Load them at app startup so the app doesn't refit on every run.
   ```python
   import joblib
   joblib.dump(scaler, 'models/scaler.pkl')
   joblib.dump(kmeans, 'models/kmeans.pkl')
   ```

3. **Handle "song not found" gracefully in the app**
   - Show a friendly message and optionally suggest the closest matching song name.

### Medium Priority

4. **Add a requirements file with pinned versions**
   - Add all runtime dependencies (`pandas`, `numpy`, `scikit-learn`, `plotly`) to `requirements.txt` with version pins for reproducibility.

5. **Add Spotify audio preview / album art**
   - Use the Spotify Web API to fetch album artwork and a 30-second preview clip for each recommendation.

6. **User filtering controls**
   - Allow the user to filter recommendations by release decade, explicit content toggle, or popularity range via Streamlit sliders and checkboxes.

### Lower Priority

7. **Deploy to Streamlit Community Cloud**
   - Push the repo to GitHub (already done), then connect it to [streamlit.io/cloud](https://streamlit.io/cloud) for a free public URL.

8. **Add unit tests**
   - Write `pytest` tests for the recommendation functions (e.g., check that the output is always a non-empty DataFrame of the requested length, that the input song itself is not returned).

9. **Explore collaborative filtering**
   - If user interaction/listen data becomes available, a collaborative filtering layer (e.g., matrix factorisation) could complement the content-based approach.

10. **Performance optimisation**
    - For the cosine-similarity search over 600k songs, consider approximate nearest-neighbour libraries such as `faiss` or `annoy` to reduce query latency from seconds to milliseconds.

---

*Last updated: March 2026*
