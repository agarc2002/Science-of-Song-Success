import pandas as pd
import numpy as np
import re
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

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

# === 3. Select feature columns ===
feature_cols = [
    "energy", "danceability", "valence", "acousticness",
    "instrumentalness", "speechiness", "loudness",
    "tempo", "duration_ms"
]
emb_cols = [col for col in meta.columns if col.startswith("emb_")]
all_features = feature_cols + emb_cols

# === 3b. Clean up embeddings if stored as strings ===
def safe_to_float(v):
    """Converts stringified lists like '[0.168]' or nested strings to float safely."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip()
        match = re.match(r"^\[+([\-0-9\.eE]+)\]+$", v)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        try:
            val = literal_eval(v)
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(val[0]) if len(val) > 0 else np.nan
            return float(val)
        except:
            return np.nan
    return np.nan

print("🧹 Deep-cleaning embedding columns (robust parser)...")
for col in emb_cols:
    meta[col] = meta[col].apply(safe_to_float)

# verify numeric conversion
non_numeric = meta[emb_cols].applymap(lambda x: isinstance(x, str)).sum().sum()
print(f"✅ Finished cleaning. Non-numeric values remaining: {non_numeric}")

# === 3c. Fix valence and tempo if still list-like ===
def strip_brackets_to_float(v):
    """Handles leftover stringified single-value lists like '[0.49225]'."""
    if isinstance(v, (float, int)):
        return v
    if isinstance(v, str):
        v = v.strip("[] ")
        try:
            return float(v)
        except:
            try:
                val = literal_eval(v)
                if isinstance(val, (list, tuple)):
                    return float(val[0])
                return float(val)
            except:
                return np.nan
    return np.nan

for col in ["valence", "tempo"]:
    meta[col] = meta[col].apply(strip_brackets_to_float)

print("✅ Fixed valence and tempo string formatting issues.")

# === Debug check for any non-numeric values ===
print("🔍 Checking for remaining non-numeric cells across all_features...")

def is_not_float(x):
    try:
        float(x)
        return False
    except:
        return True

bad_cells = {}
for col in all_features:
    bad_rows = meta[col].apply(is_not_float)
    if bad_rows.any():
        bad_cells[col] = meta.loc[bad_rows, col].head(5).tolist()

if bad_cells:
    print("⚠️ Found problematic columns and sample bad values:")
    for c, vals in bad_cells.items():
        print(f"  - {c}: {vals}")
else:
    print("✅ All features appear numeric.")

# === 4. Build numeric feature matrix ===
X = meta[all_features].fillna(0).astype(float).to_numpy()
y_pop = meta["Spotify_Track_Popularity"].fillna(0)
y_market = meta["Marketability"]

# === 5. Split into train/test sets ===
X_train, X_test, ypop_train, ypop_test, ymar_train, ymar_test = train_test_split(
    X, y_pop, y_market, test_size=0.2, random_state=42
)

# === 6. Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.transform(X)

# === 7. Train Random Forests ===
print("🚀 Training Random Forest models...")
pop_model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
market_model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

pop_model.fit(X_train_scaled, ypop_train)
market_model.fit(X_train_scaled, ymar_train)

# === 8. Evaluate Models ===
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

# === 9. Predict for all songs ===
meta["Predicted_Popularity"] = pop_model.predict(X_scaled_all)
meta["Predicted_Marketability"] = market_model.predict(X_scaled_all)

# === 10. Rescale feature scores (1–10) ===
for col in feature_cols:
    col_scaled = (meta[col] - meta[col].min()) / (meta[col].max() - meta[col].min())
    meta[f"{col}_Score"] = (col_scaled * 9 + 1).round(2)

# === 11. Find Similar Songs ===
def find_similar_songs(index, top_n=5):
    combined = meta[emb_cols].to_numpy()
    sim = cosine_similarity(combined)
    similar_idx = np.argsort(sim[index])[::-1][1 : top_n + 1]
    return meta.loc[similar_idx, ["Artist", "Track", "Predicted_Popularity", "Predicted_Marketability"]]

# === 12. Preview Results ===
cols = ["Artist", "Track", "Predicted_Popularity", "Predicted_Marketability"] + [f"{c}_Score" for c in feature_cols]
print("\n🎧 Preview of model results:")
print(meta[cols].head(10))

print("\n🔍 Similar songs to first track:")
print(find_similar_songs(0))


