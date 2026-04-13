# 🎵 Spotify Song Recommendation System

## Brief Project Description

This project aims to build a music recommendation system that suggests songs similar to a user’s input based on audio characteristics such as tempo, energy, and danceability. The system will use machine learning techniques to identify similarity between songs and provide recommendations through an interactive web interface.

---

## Team

| Name                 | Expected Responsibilities |
| :------------------- | :------------------------ |
| Ayush Khemani        | Project Co-Lead, Project Coordination, GitHub Management |
| Haseeb Raza          | Project Co-Lead, Technical Support, Team Coordination |
| Nursultan Tuleev     | Build ML Recommendation Model |
| Muna Hassan          | Data Exploration, Cleaning, EDA |
| Saidul Islam Nayan   | Build ML Recommendation Model |
| Yushay Aizaz         | Create Web Interface using Streamlit |

---

## The *Problem* behind the Project

With millions of songs available across platforms, users often struggle to discover music that matches their preferences or mood. Traditional browsing methods rely heavily on manual searching or popularity-based suggestions. This project aims to solve that problem by creating a content-based recommendation system that identifies songs with similar audio features and provides personalized suggestions automatically.

---

## Challenges

Some challenges involved in this project include:

- Handling a large dataset (600k+ songs)
- Cleaning and preprocessing real-world music data
- Selecting meaningful audio features for similarity computation
- Designing an efficient recommendation algorithm
- Integrating machine learning models with a web application
- Ensuring fast response time for recommendations

---

## Expectations

By the end of this project, we expect to deliver:

- A functional content-based music recommendation system
- A trained machine learning model capable of suggesting similar songs
- A user-friendly web interface for interaction
- A complete end-to-end data science project suitable for portfolios
- Practical experience in teamwork, GitHub collaboration, and deployment

---

## Tools & Technologies

The project will primarily use the following tools and technologies:

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Plotly
- Streamlit
- Git & GitHub


---

## How to Run the Project

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```
---

## What Has Been Built

### 1. Data Exploration & Cleaning (`notebooks/dataset_exploration.ipynb`)

The EDA notebook performs a full preprocessing pipeline on a 600k+ song Spotify CSV dataset (`data/raw/tracks.csv`):

- **Loading & inspection** — reads `tracks.csv` with Pandas and prints shape, column names, and data types.
- **Missing value analysis** — counts and visualises missing values per column; drops rows where `name` is null.
- **Duplicate removal** — detects and removes fully duplicate rows.
- **Data-type fixes**
  - Parses the string-encoded `artists` and `id_artists` columns into proper Python lists.
  - Converts `explicit` and `mode` from integers to booleans.
  - Extracts the release year from `release_date` into a new `release_year` integer column and drops the original.
- **Column normalisation** — strips whitespace, lowercases, and replaces spaces/hyphens with underscores in all column headers.
- **Output** — saves the cleaned dataframe to `data/cleaned_spotify.csv`.

---

### 2. Recommendation Models (`models/recommendations.ipynb`)

Three complementary recommendation strategies are implemented on top of the cleaned dataset:

#### a) Content-Based Recommender (Cosine Similarity)
- Selects 11 audio features: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
- Standardises the feature matrix with `StandardScaler`.
- For a given input song, computes pairwise **cosine similarity** against all tracks and returns the top-N most similar songs.

#### b) K-Means Clustering Recommender
- Runs the **Elbow Method** (SSE vs K) to guide cluster count selection.
- Trains a `MiniBatchKMeans` model (K = 7) on the full scaled feature set.
- Recommends songs by randomly sampling from the same cluster as the input track.

#### c) Hybrid Recommender (Content + Popularity)
- Fetches a broad candidate pool (50 songs) using the cosine-similarity recommender.
- Re-ranks candidates by `release_year` and `popularity` (descending) to surface recent, well-known matches.

#### Visualisation & Evaluation
- **t-SNE plot** — 2-D projection of the scaled feature space coloured by cluster label.
- **Radar chart** — overlaid Plotly polar chart comparing audio features of an input track vs. a recommended track.
- **Clustering metrics** — Silhouette Score and Davies-Bouldin Index for the K-Means solution.

---

### 3. Web Application (`app/app.py`)

A minimal **Streamlit** interface that:

- Provides a text input for the user to enter a song name.
- Returns placeholder recommendations on submit (ready to be wired up to the ML back-end).
- Runs locally with `streamlit run app/app.py`.

---

## Results

> 🚧 Full model evaluation and app screenshots will be added once the ML back-end is integrated with the Streamlit front-end.

**Current status:**
- ✅ Data pipeline complete — cleaned dataset produced from raw 600k-song CSV.
- ✅ Three recommendation algorithms implemented and tested in notebook.
- ✅ Clustering evaluated with Silhouette Score and Davies-Bouldin Index.
- ✅ Streamlit app scaffold deployed and runnable.
- ⏳ Wiring ML models into the web interface — in progress.
