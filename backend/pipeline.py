"""
Pipeline helper functions for end-to-end scoring.

These are lightweight stubs with clear extension points. They are not wired into
the Dash app yet, and network-dependent calls require proper API tokens and
client libraries. Fill in the TODOs with real API logic for production use.
"""
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests

import os, requests

YT_API_KEY = os.environ.get("YT_API_KEY")

def get_youtube_stats(song_name, artist=None):
    if not YT_API_KEY:
        return {"error": "YT_API_KEY not set", "data": {}}
    query = f"{song_name} {artist}" if artist else song_name
    try:
        # 1) search for top video
        s = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": query, "maxResults": 1, "type": "video", "key": YT_API_KEY},
            timeout=10,
        )
        s.raise_for_status()
        items = s.json().get("items", [])
        if not items:
            return {"error": "No video found", "data": {}}
        vid = items[0]["id"]["videoId"]

        # 2) fetch stats
        v = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "statistics", "id": vid, "key": YT_API_KEY},
            timeout=10,
        )
        v.raise_for_status()
        stats = v.json()["items"][0]["statistics"]
        return {
            "error": None,
            "data": {
                "video_id": vid,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            },
        }
    except Exception as e:
        return {"error": str(e), "data": {}}

# ---------- 1) Spotify lookup ----------

def get_spotify_data(song_name: str, artist: Optional[str] = None) -> Dict[str, Any]:
    """
    Search Spotify for a track, pull audio features + metadata.

    Requires a valid Spotify API token (Client Credentials or OAuth) in env:
    SPOTIFY_TOKEN=...

    Returns a dict with keys like:
    {
        "tempo": ...,
        "energy": ...,
        "danceability": ...,
        "acousticness": ...,
        "speechiness": ...,
        "instrumentalness": ...,
        "valence": ...,
        "loudness": ...,
        "duration_ms": ...,
        "popularity": ...,
        "preview_url": ...,
        "metadata": {...}
    }
    """
    token = os.environ.get("SPOTIFY_TOKEN")
    if not token:
        return {"error": "SPOTIFY_TOKEN not set", "data": {}}
    # TODO: Implement actual Spotify search + audio-features retrieval
    return {"error": "Not implemented (add Spotify API logic)", "data": {}}


# ---------- 2) YouTube stats ----------

def get_youtube_stats(song_name: str, artist: Optional[str] = None) -> Dict[str, Any]:
    """
    Search YouTube for the top video and return views/likes/comments.

    Requires a YouTube Data API key in env: YT_API_KEY=...
    """
    api_key = os.environ.get("YT_API_KEY")
    if not api_key:
        return {"error": "YT_API_KEY not set", "data": {}}

    query = f"{song_name} {artist}" if artist else song_name
    try:
        # 1) Search for the top video
        search_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "maxResults": 1,
                "type": "video",
                "key": api_key,
            },
            timeout=3,
        )
        search_resp.raise_for_status()
        items = search_resp.json().get("items", [])
        if not items:
            return {"error": "No video found", "data": {}}
        video_id = items[0]["id"]["videoId"]

        # 2) Fetch stats for that video
        stats_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "statistics", "id": video_id, "key": api_key},
            timeout=3,
        )
        stats_resp.raise_for_status()
        stats_items = stats_resp.json().get("items", [])
        if not stats_items:
            return {"error": "No stats found", "data": {"video_id": video_id}}

        stats = stats_items[0].get("statistics", {})
        data = {
            "video_id": video_id,
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        }
        return {"error": None, "data": data}
    except Exception as e:
        return {"error": str(e), "data": {}}


# ---------- 3) Lyrics via Genius ----------

def get_lyrics(song_name: str, artist: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch lyrics (or a lyrics URL) from Genius.

    Requires GENIUS_API_TOKEN in env.
    """
    token = os.environ.get("GENIUS_API_TOKEN")
    if not token:
        return {"error": "GENIUS_API_TOKEN not set", "data": {}}
    query = f"{song_name} {artist}" if artist else song_name
    try:
        resp = requests.get(
            "https://api.genius.com/search",
            params={"q": query},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json().get("response", {}).get("hits", [])
        if not hits:
            return {"error": "No lyrics found", "data": {}}
        top = hits[0]["result"]
        return {"error": None, "data": {"lyrics_url": top.get("url"), "full_result": top}}
    except Exception as e:
        return {"error": str(e), "data": {}}


# ---------- 4) MERT embedding (placeholder) ----------

def get_mert_embedding(audio_preview_url: str) -> Dict[str, Any]:
    """
    Download audio preview and run MERT to produce an embedding.
    This is a placeholder; wire in your MERT model or service here.
    """
    if not audio_preview_url:
        return {"error": "No audio preview URL provided", "data": {}}
    # TODO: Download audio, run MERT, return embedding vector/list
    return {"error": "Not implemented (add MERT inference)", "data": {}}


# ---------- 5) Preprocess features for XGB ----------

def preprocess_features(raw: Dict[str, Any],
                        scaler,
                        feature_order: Optional[list] = None) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the training feature order, then scale.

    - raw: dict of feature_name -> value
    - scaler: fitted sklearn scaler (StandardScaler)
    - feature_order: list of columns in the exact order expected by the model;
      if None, uses sorted keys from raw.

    Returns the scaled numpy array ready for model.predict.
    """
    if feature_order is None:
        feature_order = sorted(raw.keys())
    row = {k: raw.get(k, 0.0) for k in feature_order}
    df = pd.DataFrame([row]).astype(float).fillna(0.0)
    if scaler is None:
        raise ValueError("Scaler is None; load xgb_scaler.pkl before calling preprocess_features.")
    return pd.DataFrame(scaler.transform(df), columns=feature_order)

if __name__ == "__main__":
    print(get_youtube_stats("Hello", "Adele"))
