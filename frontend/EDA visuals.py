import pandas as pd
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from plotly.subplots import make_subplots

sns.set(style="whitegrid")

csv_path = r"C:\Users\edwar\Documents\all_with_metadata_embeddings_with_lyrics.csv"
meta = pd.read_csv(csv_path)
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

print(f"=æ Found {len(audio_emb_cols)} audio embeddings")
print(f"=ê Found {len(lyric_emb_cols)} lyric embeddings")

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

print(f"( Total features after adding engineered + contextual + artist + playlist proxy: {len(all_features)}")

# ================================================================
# 7. NORMALIZE ONLY AUDIO EMBEDDINGS
# ================================================================
meta[audio_emb_cols] = (
    meta[audio_emb_cols] - meta[audio_emb_cols].min()
) / (meta[audio_emb_cols].max() - meta[audio_emb_cols].min())



scaler = joblib.load(r"C:\Users\edwar\Documents\scaler.pkl")
X = meta[all_features].astype(float)
X_scaled_all = scaler.transform(X)

pop_model = XGBRegressor()
pop_model.load_model(r"C:\Users\edwar\Documents\popularity_model.json")

market_model = XGBRegressor()
market_model.load_model(r"C:\Users\edwar\Documents\marketability_model.json")

meta["Predicted_Popularity"] = pop_model.predict(X_scaled_all)
meta["Predicted_Marketability"] = market_model.predict(X_scaled_all)


#scatter plot for Predicted vs Actual
def plot_pred_vs_actual_combined(meta):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted vs Actual Popularity", "Predicted vs Actual Marketability"),
        horizontal_spacing=0.12
    )


    fig.add_trace(
        go.Scatter(
            x=meta["Spotify_Track_Popularity"],
            y=meta["Predicted_Popularity"],
            mode="markers",
            marker=dict(size=6, color=meta["Predicted_Popularity"], colorscale="Viridis", opacity=0.6),
            name="Popularity"
        ),
        row=1, col=1
    )

    pop_fit = px.scatter(
        meta,
        x="Spotify_Track_Popularity",
        y="Predicted_Popularity",
        trendline="ols"
    )
    trend_pop = pop_fit.data[1]   # OLS line
    fig.add_trace(trend_pop.update(showlegend=False), row=1, col=1)

    fig.update_xaxes(title_text="Popularity", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Popularity", row=1, col=1)


    fig.add_trace(
        go.Scatter(
            x=meta["Marketability"],
            y=meta["Predicted_Marketability"],
            mode="markers",
            marker=dict(size=6, color=meta["Predicted_Marketability"], colorscale="Plasma", opacity=0.6),
            name="Marketability"
        ),
        row=1, col=2
    )

    mark_fit = px.scatter(
        meta,
        x="Marketability",
        y="Predicted_Marketability",
        trendline="ols"
    )
    trend_mark = mark_fit.data[1]
    fig.add_trace(trend_mark.update(showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="Marketability", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Marketability", row=1, col=2)

    fig.update_layout(
        title="Predicted vs Actual: Popularity & Marketability",
        height=600,
        width=1200,
        coloraxis_colorbar=dict(title="Value"),
        showlegend=False,
        template="plotly_white"
    )

    return fig


fig = plot_pred_vs_actual_combined(meta)
fig.show()
fig.write_html("predicted_relationship.html")
print("✅ Saved : predicted_relationship.html")


#histogram
def make_marketability_histogram(meta):

    metric_map = {
        "Marketability": "Marketability",
        "Predicted Marketability": "Predicted_Marketability",
        "Popularity": "Spotify_Track_Popularity",
        "Predicted Popularity": "Predicted_Popularity"
    }

    default_col = "Marketability"

    fig = px.histogram(
        meta,
        x=default_col,
        nbins=50,
        opacity=0.85,
        color_discrete_sequence=["#4C78A8"],
        title="Popularity & Marketability Score Distributions"
    )


    fig.update_layout(
        bargap=0.02,
        plot_bgcolor="rgba(245,245,255,1)",
        paper_bgcolor="white",
        title_font_size=20,
        xaxis_title=default_col,
        yaxis_title="Count"
    )

    buttons = []
    for label, colname in metric_map.items():
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"x": [meta[colname]]},  # update histogram data
                    {
                        "title": f"Histogram — {label}",
                        "xaxis": {"title": label},
                        "marker": {
                            "color": meta[colname],
                            "colorscale": "Viridis",  # 🔥 beautiful colorscale
                        }
                    }
                ]
            )
        )


    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.12,
                y=1.15,
                xanchor="left"
            )
        ]
    )

    return fig

fig = make_marketability_histogram(meta)
fig.write_html("histogram.html")
print("✅ Saved:histogram.html")
fig.show()




# 2D AUDIO MAP
def make_audio_pca_dropdown(meta):

    audio_emb_cols = [c for c in meta.columns if c.startswith("emb_")]
    if len(audio_emb_cols) == 0:
        raise ValueError("No audio embedding columns found (emb_*)")

    
    meta[audio_emb_cols] = meta[audio_emb_cols].apply(
        lambda s: pd.to_numeric(
            s.astype(str).str.replace(r'[\[\]]','', regex=True),
            errors="coerce"
        )
    ).fillna(0)

    
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(meta[audio_emb_cols])

    audio_pca_df = pd.DataFrame({
        "Audio Dimension 1": reduced[:, 0],
        "Audio Dimension 2": reduced[:, 1],
        "Artist": meta["Artist"],
        "Track": meta["Track"],
        "Popularity": meta["Spotify_Track_Popularity"],
        "Predicted_Popularity": meta["Predicted_Popularity"],
        "Marketability": meta["Marketability"],
        "Predicted_Marketability": meta["Predicted_Marketability"]
    })

    
    fig = px.scatter(
        audio_pca_df,
        x="Audio Dimension 1",
        y="Audio Dimension 2",
        color="Popularity",
        hover_data=["Artist", "Track"],
        title="2D Audio PCA Map",
        color_continuous_scale="Viridis",
        width=1000,
        height=700,
    )

    
    def color_button(label, column):
        return dict(
            label=label,
            method="update",
            args=[
                {
                    "marker": {
                        "color": audio_pca_df[column],
                        "colorscale": "Viridis",
                        "coloraxis": "coloraxis"
                    }
                },
                {
                    "coloraxis": {
                        "colorscale": "Viridis",
                        "colorbar": {"title": label}
                    }
                }
            ]
        )

    dropdown_buttons = [
        color_button("Popularity", "Popularity"),
        color_button("Predicted Popularity", "Predicted_Popularity"),
        color_button("Marketability", "Marketability"),
        color_button("Predicted Marketability", "Predicted_Marketability"),
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                x=1.15,
                y=1.0,
                showactive=True
            )
        ]
    )

    return fig


fig = make_audio_pca_dropdown(meta)
fig.show()
fig.write_html("audio_pca.html")
print("✅ Saved audio_pca.html")
     

#heatmap for audio
def make_triangular_acoustic_heatmap_numbers(meta):
    acoustic_cols = [
        "energy","danceability","valence","acousticness",
        "instrumentalness","speechiness","loudness","tempo"
    ]

    df = meta[acoustic_cols].copy()
    corr = df.corr().round(2)

    
    mask = np.tril(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(mask)     

    
    text_matrix = corr_masked.astype(str)
    text_matrix = text_matrix.mask(mask == False, "")  

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_masked.values,
            x=acoustic_cols,
            y=acoustic_cols,
            text=text_matrix.values,
            texttemplate="%{text}",          
            textfont={"size": 12},
            colorscale="RdBu",
            reversescale=True,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Correlation"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{text}<extra></extra>"
        )
    )

    # 4. Layout styling
    fig.update_layout(
        title="Triangular Acoustic Feature Correlation Heatmap (with values)",
        title_font=dict(size=22),
        width=950,
        height=900,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=60, t=80, b=60)
    )

    return fig



fig = make_triangular_acoustic_heatmap_numbers(meta)
fig.show()
fig.write_html("triangular_heatmap.html")
print("✅triangular_heatmap.html")

#matrix
def make_acoustic_feature_matrix(meta):
    
    acoustic_cols = [
        "energy", "danceability", "valence", "acousticness",
        "instrumentalness", "speechiness", "loudness", "tempo"
    ]

    df = meta[acoustic_cols].copy()

    
    rename_map = {
        "energy": "energy",
        "danceability": "dance",
        "valence": "valence",
        "acousticness": "acoustic",
        "instrumentalness": "instr",
        "speechiness": "speech",
        "loudness": "loud",
        "tempo": "tempo"
    }

    df_short = df.rename(columns=rename_map)

    fig = px.scatter_matrix(
        df_short,
        dimensions=list(df_short.columns),
        title="Acoustic Feature Relationships",
        height=900,
        width=900
    )

    fig.update_traces(diagonal_visible=False, opacity=0.55)

    # Small style tweak so it stays readable
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=80)
    )

    return fig

fig1 = make_acoustic_feature_matrix(meta)
fig1.write_html("matrix.html")
print("✅ Saved: matrix.html")


#Engagement vs Audio Features
def make_engagement_vs_audio_sidepanel(meta):

    import plotly.express as px

    
    engagement_metrics = {
        "Engagement Rate": "engagement_rate",
        "Predicted Popularity": "Predicted_Popularity",
        "Predicted Marketability": "Predicted_Marketability"
    }

    
    audio_cols = [
        "energy", "valence", "acousticness",
        "instrumentalness", "speechiness",
        "loudness", "tempo"
    ]

    
    fig = px.scatter(
        meta,
        x="energy",
        y="Predicted_Popularity",
        opacity=0.6,
        title="Engagement vs Audio Feature",
        height=850,
        width=1100,
        hover_data=["Artist", "Track", "Views", "Likes"]
    )

    
    fig.update_layout(
        margin=dict(l=60, r=300, t=80, b=60)
    )

    
    PANEL_X = 1.10
    AUDIO_Y = 0.80
    ENGAGE_Y = 0.45

    
    def audio_button(feat):
        return dict(
            label=feat,
            method="update",
            args=[
                {
                    "x": [meta[feat]],
                    "hovertemplate": f"<b>{feat}</b>: %{{x}}<br>" +
                                     "<b>%{yaxis.title.text}</b>: %{y}<br>" +
                                     "<b>Artist</b>: %{customdata[0]}<br>" +
                                     "<b>Track</b>: %{customdata[1]}<br>" +
                                     "<extra></extra>"
                },
                {"xaxis": {"title": feat}}
            ]
        )

    def engage_button(label, col):
        return dict(
            label=label,
            method="update",
            args=[
                {
                    "y": [meta[col]],
                    "hovertemplate": f"<b>%{{xaxis.title.text}}</b>: %{{x}}<br>"
                                     f"<b>{label}</b>: %{{y}}<br>"
                                     "<b>Artist</b>: %{customdata[0]}<br>"
                                     "<b>Track</b>: %{customdata[1]}<br>"
                                     "<extra></extra>"
                },
                {"yaxis": {"title": label}}
            ]
        )

    
    fig.update_layout(
        updatemenus=[

            
            dict(
                type="dropdown",
                buttons=[audio_button(f) for f in audio_cols],
                x=PANEL_X, y=AUDIO_Y,
                xanchor="left", yanchor="middle"
            ),

            
            dict(
                type="dropdown",
                buttons=[engage_button(lbl, col) for lbl, col in engagement_metrics.items()],
                x=PANEL_X, y=ENGAGE_Y,
                xanchor="left", yanchor="middle"
            )
        ]
    )

   
    fig.update_layout(
        annotations=[
            dict(
                text="<b>Audio Feature</b>",
                x=PANEL_X +.17, y=AUDIO_Y + 0.085,
                xref="paper", yref="paper",
                showarrow=False, font=dict(size=15)
            ),
            dict(
                text="<b>Engagement Metric</b>",
                x=PANEL_X+.23, y=ENGAGE_Y + 0.06,
                xref="paper", yref="paper",
                showarrow=False, font=dict(size=15)
            )
        ]
    )

    return fig

fig = make_engagement_vs_audio_sidepanel(meta)
fig.write_html("engagement_vs_audio.html")
print("✅ Saved: engagement_vs_audio.html")



