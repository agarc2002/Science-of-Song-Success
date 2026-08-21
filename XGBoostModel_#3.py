import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# ================================================================
# 1. LOAD ENRICHED DATASET (WITH LYRICS)
# ================================================================
csv_path = r"C:\Users\rohan\Downloads\STA 160\all_with_metadata_embeddings_with_lyrics.csv"
meta = pd.read_csv(csv_path)
print(f"✅ Loaded enriched dataset with shape: {meta.shape}")

# ================================================================
# 2. COMPUTE MARKETABILITY SCORE
# ================================================================
meta["Marketability"] = (
    0.4 * meta["Spotify_Track_Popularity"].fillna(0)
    + 0.3 * np.log1p(meta["Views"].fillna(0))
    + 0.2 * (meta["Likes"].fillna(0) / (meta["Views"].fillna(1) + 1) * 100)
    + 0.05 * meta["energy"].fillna(0)
    + 0.05 * meta["danceability"].fillna(0)
).clip(0, 100)

# ================================================================
# 3. FEATURE GROUPS
# ================================================================
audio_cols = [
    "energy","danceability","valence","acousticness",
    "instrumentalness","speechiness","loudness","tempo","duration_ms"
]

audio_emb_cols = [c for c in meta.columns if c.startswith("emb_")]
lyric_emb_cols = [c for c in meta.columns if c.startswith("lyric_emb_")]

print(f"🟦 Found {len(audio_emb_cols)} audio embeddings")
print(f"🟪 Found {len(lyric_emb_cols)} lyric embeddings")

all_features = audio_cols + audio_emb_cols + lyric_emb_cols

# ================================================================
# 4. FORCE NUMERIC VALUES
# ================================================================
def force_to_float(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r'[\[\]]','',regex=True),
        errors='coerce'
    )

meta[all_features] = meta[all_features].apply(force_to_float).fillna(0)

# ================================================================
# 5. ENGINEERED FEATURES
# ================================================================
meta["energy_dance"] = meta["energy"] * meta["danceability"]
meta["valence_energy"] = meta["valence"] * meta["energy"]
meta["valence_dance"] = meta["valence"] * meta["danceability"]
meta["tempo_energy"] = meta["tempo"] * meta["energy"]
meta["energy_valence_ratio"] = meta["energy"] / (meta["valence"] + 1e-5)
meta["speech_music_ratio"] = meta["speechiness"] / (meta["instrumentalness"] + 1e-5)
meta["tempo_z"] = (meta["tempo"] - meta["tempo"].mean()) / meta["tempo"].std()

engineered_cols = [
    "energy_dance","valence_energy","valence_dance",
    "tempo_energy","energy_valence_ratio","speech_music_ratio","tempo_z"
]

all_features += engineered_cols

# ================================================================
# 6. CONTEXTUAL + ARTIST + PLAYLIST-PROXY FEATURES
# ================================================================
for col in ["Views","Likes","Comments"]:
    meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)

meta["like_view_ratio"] = meta["Likes"] / (meta["Views"] + 1)
meta["comment_view_ratio"] = meta["Comments"] / (meta["Views"] + 1)
meta["engagement_rate"] = (meta["Likes"] + meta["Comments"]) / (meta["Views"] + 1)

meta["log_views"] = np.log1p(meta["Views"])
meta["log_likes"] = np.log1p(meta["Likes"])
meta["log_comments"] = np.log1p(meta["Comments"])

if "Artist" in meta.columns:
    meta["artist_avg_popularity"] = (
        meta.groupby("Artist")["Spotify_Track_Popularity"]
        .transform("mean")
        .fillna(0)
    )
else:
    meta["artist_avg_popularity"] = 0

if "UploadDate" in meta.columns:
    meta["UploadDate"] = pd.to_datetime(meta["UploadDate"], errors="coerce")
    ref_date = meta["UploadDate"].max()
    meta["days_since_upload"] = (ref_date - meta["UploadDate"]).dt.days.fillna(0)
    meta["recency_score"] = np.exp(-meta["days_since_upload"] / 365)
else:
    meta["recency_score"] = 1

meta["social_influence"] = (
    0.4 * meta["log_views"]
    + 0.3 * meta["log_likes"]
    + 0.2 * meta["log_comments"]
    + 0.1 * meta["artist_avg_popularity"]
)

context_cols = [
    "like_view_ratio","comment_view_ratio","engagement_rate",
    "log_views","log_likes","log_comments",
    "artist_avg_popularity","social_influence","recency_score"
]

# ---------- NEW: ARTIST-LEVEL FEATURES ----------
if "Artist" in meta.columns:
    # how many tracks this artist has in the dataset
    meta["artist_track_count"] = meta.groupby("Artist")["Track"].transform("count")

    # average views / likes for this artist across all their songs
    meta["artist_mean_views"] = meta.groupby("Artist")["Views"].transform("mean")
    meta["artist_mean_likes"] = meta.groupby("Artist")["Likes"].transform("mean")

    # overall engagement across artist catalog
    artist_views_sum = meta.groupby("Artist")["Views"].transform("sum") + 1
    artist_likes_sum = meta.groupby("Artist")["Likes"].transform("sum")
    meta["artist_engagement"] = artist_likes_sum / artist_views_sum
else:
    meta["artist_track_count"] = 0
    meta["artist_mean_views"] = 0
    meta["artist_mean_likes"] = 0
    meta["artist_engagement"] = 0

artist_cols = [
    "artist_track_count",
    "artist_mean_views",
    "artist_mean_likes",
    "artist_engagement"
]

# ---------- NEW: PLAYLIST MOMENTUM PROXY ----------
meta["playlist_proxy"] = (
    0.5 * meta["log_views"] +
    0.3 * meta["log_likes"] +
    0.1 * meta["log_comments"] +
    0.1 * meta["recency_score"]
)

playlist_cols = ["playlist_proxy"]

# add everything to feature list
all_features += context_cols + artist_cols + playlist_cols

print(f"✨ Total features after adding engineered + contextual + artist + playlist proxy: {len(all_features)}")

# ================================================================
# 7. NORMALIZE ONLY AUDIO EMBEDDINGS
# ================================================================
meta[audio_emb_cols] = (
    meta[audio_emb_cols] - meta[audio_emb_cols].min()
) / (meta[audio_emb_cols].max() - meta[audio_emb_cols].min())

# ================================================================
# 8. TRAIN / TEST SPLIT
# ================================================================
X = meta[all_features].astype(float)
y_pop = meta["Spotify_Track_Popularity"].fillna(0)
y_market = meta["Marketability"]

X_train, X_test, ypop_train, ypop_test = train_test_split(
    X, y_pop, test_size=0.2, random_state=42
)
_, _, ymar_train, ymar_test = train_test_split(
    X, y_market, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.transform(X)

# ================================================================
# 9. XGBOOST HYPERPARAMETERS
# ================================================================
xgb_params = dict(
    n_estimators=2500,
    learning_rate=0.008,
    max_depth=12,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=0.7,
    reg_alpha=0.15,
    min_child_weight=1,
    gamma=0.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

early_stop = EarlyStopping(rounds=200, save_best=True)

# ================================================================
# 10. TRAIN MODELS
# ================================================================
print("\n🚀 Training Popularity Model...")
pop_model = XGBRegressor(**xgb_params)
pop_model.fit(X_train_scaled, ypop_train)

print("\n🚀 Training Marketability Model...")
market_model = XGBRegressor(**xgb_params)
market_model.fit(X_train_scaled, ymar_train)

# ================================================================
# 11. EVALUATE MODELS
# ================================================================
def evaluate(name, model, X_t, y_t):
    pred = model.predict(X_t)
    r2 = r2_score(y_t, pred)
    mae = mean_absolute_error(y_t, pred)
    print(f"\n📊 {name} Model Performance:")
    print(f"   R²:  {r2:.3f}")
    print(f"   MAE: {mae:.3f}")
    return pred

pop_pred = evaluate("Popularity", pop_model, X_test_scaled, ypop_test)
mark_pred = evaluate("Marketability", market_model, X_test_scaled, ymar_test)

# ================================================================
# 12. FEATURE IMPORTANCE PLOTS
# ================================================================
def plot_importance(model, name):
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Gain": list(importance.values())
    }).sort_values("Gain", ascending=False).head(15)

    plt.figure(figsize=(8,5))
    plt.barh(imp_df["Feature"], imp_df["Gain"])
    plt.gca().invert_yaxis()
    plt.title(f"Top Features — {name}")
    plt.show()

plot_importance(pop_model, "Popularity")
plot_importance(market_model, "Marketability")

# ================================================================
# 13. PREDICT FOR ENTIRE DATASET
# ================================================================
meta["Predicted_Popularity"] = pop_model.predict(X_scaled_all)
meta["Predicted_Marketability"] = market_model.predict(X_scaled_all)

# ================================================================
# 14. HYBRID SONG SIMILARITY MATRIX (Audio + Lyrics + Predictions)
# ================================================================
print("\n🎧 Building HYBRID similarity matrix...")

# ---- 1) AUDIO feature subset
sim_audio_cols = ["energy","danceability","valence","tempo","loudness"]
audio_features = meta[sim_audio_cols].fillna(0).values

# ---- 2) AUDIO embeddings (PCA reduced)
pca_audio = PCA(n_components=40, random_state=42)
audio_emb_reduced = pca_audio.fit_transform(meta[audio_emb_cols].fillna(0))

# ---- 3) LYRIC embeddings (PCA reduced)
lyric_cols = [c for c in meta.columns if c.startswith("lyric_emb_")]
if len(lyric_cols) > 0:
    pca_lyrics = PCA(n_components=40, random_state=42)
    lyric_emb_reduced = pca_lyrics.fit_transform(meta[lyric_cols].fillna(0))
else:
    lyric_emb_reduced = np.zeros((len(meta), 40))

# ---- 4) MODEL PREDICTIONS
pred_matrix = np.vstack([
    meta["Predicted_Popularity"].values,
    meta["Predicted_Marketability"].values
]).T  # shape (N, 2)

# ---- 5) BUILD FINAL HYBRID EMBEDDING
hybrid_embedding = np.concatenate([
    audio_features * 0.25,        # 25% weight
    audio_emb_reduced * 0.35,     # 35% weight
    lyric_emb_reduced * 0.25,     # 25% weight
    pred_matrix * 0.15            # 15% weight
], axis=1)

# ---- 6) COSINE SIMILARITY
sim_matrix = cosine_similarity(hybrid_embedding)

def find_similar_songs(index, top_n=5):
    sim_scores = sim_matrix[index]
    idx = np.argsort(sim_scores)[::-1][1:top_n+1]
    return meta.loc[idx, ["Artist","Track","Predicted_Popularity","Predicted_Marketability"]]

print("\n🔍 Top 5 hybrid-similar songs to first track:")
print(find_similar_songs(0))

# ================================================================
# DONE
# ================================================================
print("\n🎉 Model training, evaluation, and similarity generation complete!")
