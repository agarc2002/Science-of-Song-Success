# train_xgb_models.py (adjusted)
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump
from xgboost import XGBRegressor

CSV_PATH = Path("merged_no_embeddings.csv")

# Use the labels aligned with XGB3_model.py logic
TARGET_POP_COL = "Spotify_Track_Popularity"   # true popularity label
TARGET_MKT_COL = "Marketability"              # engineered marketability score

# Exclude non-feature columns; keep only numeric features
NON_FEATURE_COLS = {
    TARGET_POP_COL, TARGET_MKT_COL,
    "Predicted_Popularity", "Predicted_Marketability",
    "Artist","Track","Album","Url_spotify","Url_youtube","Title","Channel",
    "Licensed","official_video","video_id","Track_URL_Spotify","Lyrics",
    "filename","Uri"  # add/remove as needed
}

assert CSV_PATH.exists(), f"Feature table not found: {CSV_PATH}"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TARGET_POP_COL, TARGET_MKT_COL])
print(f"Loaded {len(df):,} rows")

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
print(f"Using {len(feature_cols)} feature columns")

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_pop = pd.to_numeric(df[TARGET_POP_COL], errors="coerce")
y_mkt = pd.to_numeric(df[TARGET_MKT_COL], errors="coerce")
mask = y_pop.notna() & y_mkt.notna()
X, y_pop, y_mkt = X.loc[mask], y_pop.loc[mask], y_mkt.loc[mask]

X_train, X_valid, y_pop_train, y_pop_valid = train_test_split(X, y_pop, test_size=0.2, random_state=42)
_, _, y_mkt_train, y_mkt_valid = train_test_split(X, y_mkt, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_valid_s = scaler.transform(X_valid)
dump(scaler, "xgb_scaler.pkl")

xgb_params = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

pop_model = XGBRegressor(**xgb_params).fit(X_train_s, y_pop_train)
y_pop_pred = pop_model.predict(X_valid_s)
rmse_pop = mean_squared_error(y_pop_valid, y_pop_pred) ** 0.5
print(f"Popularity – R² {r2_score(y_pop_valid, y_pop_pred):.3f}, RMSE {rmse_pop:.3f}")
dump(pop_model, "xgb_popularity.pkl")

mkt_params = {**xgb_params, "random_state": 43}
mkt_model = XGBRegressor(**mkt_params).fit(X_train_s, y_mkt_train)
y_mkt_pred = mkt_model.predict(X_valid_s)
rmse_mkt = mean_squared_error(y_mkt_valid, y_mkt_pred) ** 0.5
print(f"Marketability – R² {r2_score(y_mkt_valid, y_mkt_pred):.3f}, RMSE {rmse_mkt:.3f}")
dump(mkt_model, "xgb_marketability.pkl")

print("Saved: xgb_scaler.pkl, xgb_popularity.pkl, xgb_marketability.pkl")
