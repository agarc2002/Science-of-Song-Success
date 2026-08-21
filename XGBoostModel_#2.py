import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# === 1. Load dataset ===
csv_path = r"C:\Users\rohan\Downloads\STA 160\all_with_metadata_embeddings.csv"
meta = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with shape: {meta.shape}")

# === 2. Compute Marketability ===
meta["Marketability"] = (
    0.4 * meta["Spotify_Track_Popularity"].fillna(0)
    + 0.3 * np.log1p(meta["Views"].fillna(0))
    + 0.2 * (meta["Likes"].fillna(0) / (meta["Views"].fillna(1) + 1) * 100)
    + 0.05 * meta["energy"].fillna(0)
    + 0.05 * meta["danceability"].fillna(0)
).clip(0, 100)

# === 3. Base audio features ===
feature_cols = [
    "energy","danceability","valence","acousticness",
    "instrumentalness","speechiness","loudness","tempo","duration_ms"
]
emb_cols = [c for c in meta.columns if c.startswith("emb_")]
all_features = feature_cols + emb_cols

# === 4. Clean numeric columns ===
def force_to_float(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[\[\]]','',regex=True), errors='coerce')
meta[all_features] = meta[all_features].apply(force_to_float).fillna(0)

# === 5. Feature engineering (acoustic interactions) ===
meta["energy_dance"] = meta["energy"] * meta["danceability"]
meta["valence_energy"] = meta["valence"] * meta["energy"]
meta["valence_dance"] = meta["valence"] * meta["danceability"]
meta["tempo_energy"] = meta["tempo"] * meta["energy"]
meta["energy_valence_ratio"] = meta["energy"] / (meta["valence"] + 1e-5)
meta["speech_music_ratio"] = meta["speechiness"] / (meta["instrumentalness"] + 1e-5)
meta["tempo_z"] = (meta["tempo"] - meta["tempo"].mean()) / meta["tempo"].std()

engineered_cols = [
    "energy_dance","valence_energy","valence_dance","tempo_energy",
    "energy_valence_ratio","speech_music_ratio","tempo_z"
]
all_features += engineered_cols

# === 6. Add contextual features ===
for col in ["Views","Likes","Comments"]:
    if col in meta.columns:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)

meta["like_view_ratio"] = meta["Likes"] / (meta["Views"] + 1)
meta["comment_view_ratio"] = meta["Comments"] / (meta["Views"] + 1)
meta["engagement_rate"] = (meta["Likes"] + meta["Comments"]) / (meta["Views"] + 1)

meta["log_views"] = np.log1p(meta["Views"])
meta["log_likes"] = np.log1p(meta["Likes"])
meta["log_comments"] = np.log1p(meta["Comments"])

if "Artist" in meta.columns:
    meta["artist_avg_popularity"] = meta.groupby("Artist")["Spotify_Track_Popularity"].transform("mean").fillna(0)

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
all_features += context_cols

print(f"✅ Added {len(context_cols)} contextual features. Total features: {len(all_features)}")

# === 7. Normalize embeddings ===
meta[emb_cols] = (meta[emb_cols] - meta[emb_cols].min()) / (meta[emb_cols].max() - meta[emb_cols].min())

# === 8. Prepare matrices ===
X = meta[all_features].astype(float)
y_pop = meta["Spotify_Track_Popularity"].fillna(0)
y_market = meta["Marketability"]

X_train, X_test, ypop_train, ypop_test = train_test_split(X, y_pop, test_size=0.2, random_state=42)
_, _, ymar_train, ymar_test = train_test_split(X, y_market, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.transform(X)

# === 9. Train XGBoost ===
xgb_params = dict(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=12,
    subsample=0.95,
    colsample_bytree=0.9,
    reg_lambda=0.5,
    reg_alpha=0.1,
    min_child_weight=1,
    gamma=0.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

early_stop = EarlyStopping(rounds=150, save_best=True)

print("🚀 Training popularity model...")
pop_model = XGBRegressor(**xgb_params)
pop_model.fit(X_train_scaled, ypop_train)

print("🚀 Training marketability model...")
market_model = XGBRegressor(**xgb_params)
market_model.fit(X_train_scaled, ymar_train)

# === 10. Evaluate ===
def evaluate(name, model, X_t, y_t):
    pred = model.predict(X_t)
    r2 = r2_score(y_t, pred)
    mae = mean_absolute_error(y_t, pred)
    print(f"\n{name} Model Results:")
    print(f"  R²:  {r2:.3f}")
    print(f"  MAE: {mae:.3f}")
    return pred

pop_pred = evaluate("Popularity", pop_model, X_test_scaled, ypop_test)
mark_pred = evaluate("Marketability", market_model, X_test_scaled, ymar_test)

# === 11. Feature importance visualization ===
def plot_importance(model, name):
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    imp_df = pd.DataFrame({'Feature': list(importance.keys()), 'Gain': list(importance.values())})
    imp_df = imp_df.sort_values('Gain', ascending=False).head(15)
    plt.figure(figsize=(8,5))
    plt.barh(imp_df['Feature'], imp_df['Gain'])
    plt.gca().invert_yaxis()
    plt.title(f"Top Features - {name}")
    plt.show()

plot_importance(pop_model, "Popularity")
plot_importance(market_model, "Marketability")

# === 12. Predict all ===
meta["Predicted_Popularity"] = pop_model.predict(X_scaled_all)
meta["Predicted_Marketability"] = market_model.predict(X_scaled_all)

# === 13. Similarity matrix ===
print("\n🎧 Building similarity matrix (audio features + PCA-reduced embeddings)...")
similarity_features = ["energy","danceability","valence","tempo","loudness"]
pca = PCA(n_components=50, random_state=42)
emb_reduced = pca.fit_transform(meta[emb_cols].fillna(0))

# Weighted sonic space (audio + embeddings)
sonic_space = np.concatenate([
    meta[similarity_features].to_numpy() * 0.4,
    emb_reduced * 0.6
], axis=1)
sim_matrix = cosine_similarity(sonic_space)

def find_similar_songs(index, top_n=5):
    sim_scores = sim_matrix[index]
    similar_idx = np.argsort(sim_scores)[::-1][1:top_n+1]
    return meta.loc[similar_idx, ["Artist","Track","Predicted_Popularity","Predicted_Marketability"]]

# === 14. Preview ===
cols = ["Artist","Track","Predicted_Popularity","Predicted_Marketability"]+[f"{c}" for c in feature_cols[:6]]
print("\n🎵 Preview of predictions:")
print(meta[cols].head(10))

print("\n🔍 Top 5 similar songs to first track:")
print(find_similar_songs(0))
