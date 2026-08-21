import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# === 1. Load merged dataset ===
csv_path = r"C:\Users\rohan\Downloads\STA 160\all_with_metadata_embeddings.csv"
meta = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with shape: {meta.shape}")

# === 2. Compute Marketability Score ===
meta["Marketability"] = (
    0.4 * meta["Spotify_Track_Popularity"].fillna(0)
    + 0.3 * np.log1p(meta["Views"].fillna(0))
    + 0.2 * (meta["Likes"].fillna(0) / (meta["Views"].fillna(1) + 1) * 100)
    + 0.05 * meta["energy"].fillna(0)
    + 0.05 * meta["danceability"].fillna(0)
)
meta["Marketability"] = meta["Marketability"].clip(0, 100)

# === 3. Feature selection ===
feature_cols = [
    "energy", "danceability", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness",
    "tempo", "duration_ms"
]
emb_cols = [col for col in meta.columns if col.startswith("emb_")]
all_features = feature_cols + emb_cols

# === 4. Prepare matrices ===
X = meta[all_features].fillna(0).astype(float)
y_pop = meta["Spotify_Track_Popularity"].fillna(0)
y_market = meta["Marketability"]

# Split train/test
X_train, X_test, ypop_train, ypop_test, ymar_train, ymar_test = train_test_split(
    X, y_pop, y_market, test_size=0.2, random_state=42
)

# === 5. Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.transform(X)

# === 6. Train XGBoost models ===
print("🚀 Training XGBoost models... (this may take 5–10 minutes)")

xgb_params = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    verbosity=0
)

pop_model = XGBRegressor(**xgb_params)
market_model = XGBRegressor(**xgb_params)

pop_model.fit(X_train_scaled, ypop_train)
market_model.fit(X_train_scaled, ymar_train)

# === 7. Evaluate models ===
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

# === 8. Predict all songs ===
meta["Predicted_Popularity"] = pop_model.predict(X_scaled_all)
meta["Predicted_Marketability"] = market_model.predict(X_scaled_all)

# === 9. Feature scores (1–10 scale) ===
for col in feature_cols:
    col_scaled = (meta[col] - meta[col].min()) / (meta[col].max() - meta[col].min())
    meta[f"{col}_Score"] = (col_scaled * 9 + 1).round(2)

# === 10. Song similarity (audio-based only) ===
print("\n🎧 Building similarity matrix (audio features + PCA-reduced embeddings)...")

# use key audio features + PCA on embeddings
similarity_features = ["energy", "danceability", "valence", "tempo", "loudness"]
pca = PCA(n_components=50, random_state=42)
emb_reduced = pca.fit_transform(meta[emb_cols].fillna(0))
sonic_space = np.concatenate([meta[similarity_features].to_numpy(), emb_reduced], axis=1)

# cosine similarity in this reduced sonic space
sim_matrix = cosine_similarity(sonic_space)

def find_similar_songs(index, top_n=5):
    sim_scores = sim_matrix[index]
    similar_idx = np.argsort(sim_scores)[::-1][1 : top_n + 1]
    return meta.loc[similar_idx, ["Artist", "Track", "Predicted_Popularity", "Predicted_Marketability"]]

# === 11. Preview ===
cols = ["Artist", "Track", "Predicted_Popularity", "Predicted_Marketability"] + [f"{c}_Score" for c in feature_cols]
print("\n🎵 Preview of model predictions:")
print(meta[cols].head(10))

print("\n🔍 Top 5 similar songs to first track:")
print(find_similar_songs(0))

