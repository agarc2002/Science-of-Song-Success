# merge_tracks.py
# Strong-normalization merge (features → SyData) + dedupe + optional fuzzy auto-accept.

import re
import unicodedata
from pathlib import Path
from typing import Optional, List

import pandas as pd
from rapidfuzz import fuzz

# ========= CONFIG =========
# Update paths if needed:
FEAT_PATH = Path("/Users/alexg20/Downloads/STA160_project/audio_features_full_all_20251025_1205.csv")
SY_PATH   = Path("/Users/alexg20/Downloads/STA160_project/SyData.csv")

OUT_MERGED_EXACT_RAW   = Path("/Users/alexg20/Downloads/STA160_project/merged_exact_strong_normalization.csv")
OUT_MERGED_EXACT_CANON = Path("/Users/alexg20/Downloads/STA160_project/merged_exact_canonical.csv")
OUT_REVIEW_CANDS       = Path("/Users/alexg20/Downloads/STA160_project/review_candidates.csv")
OUT_FINAL_WITH_FUZZY   = Path("/Users/alexg20/Downloads/STA160_project/merged_final_with_fuzzy95.csv")

# Dedup preference: try these columns (in order) if they exist to keep the "best" SyData row per filename.
POPULARITY_COLUMNS: List[str] = ["Spotify_Track_Popularity", "Spotify_Popularity", "Popularity"]

# Fuzzy settings
BUILD_FUZZY_REVIEW = True
FUZZ_THRESHOLD = 90            # review candidates threshold
MAX_CANDIDATES_PER_ROW = 3     # top-N per unmatched features row
LENGTH_WINDOW = 3              # candidate key length tolerance
BLOCK_BY_FIRST_CHAR = True     # simple blocker to speed up fuzzy

# Auto-accept very high-confidence fuzzy matches and append to final output
AUTO_ACCEPT_FUZZY = False
AUTO_ACCEPT_THRESHOLD = 95     # only fuzzy >= this get auto-added
# ==========================


def strip_accents_text(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


NOISE_PATTERNS = [
    r'\bofficial\s*video\b', r'\blyrics?\b', r'\baudio\b', r'\bvisualizer\b',
    r'\bremaster(?:ed)?\b', r'\bexplicit\b', r'\blive\b', r'\bhd\b', r'\bclip\b',
    r'\bvideo\b', r'\bkaraoke\b', r'\binstrumental\b', r'\bradio\s*edit\b',
    r'\bclean\b', r'\bprod\.?\b', r'\bfeat\.?\b', r'\bft\.?\b', r'\bx\b'
]
NOISE_RE = re.compile('|'.join(NOISE_PATTERNS), flags=re.IGNORECASE)


def clean_series_base(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str)
    # remove audio extensions
    s = s.str.replace(r'\.(opus|mp3|wav|m4a)$', '', regex=True, flags=re.IGNORECASE)
    # remove bracketed content
    s = s.str.replace(r'[\[\(\{].*?[\]\)\}]', ' ', regex=True)
    # normalize separators and punctuation
    s = s.str.replace(r'[•|–—_]', ' ', regex=True)
    s = s.str.replace(r'&', ' and ', regex=True)
    s = s.str.replace(r'[^0-9a-zA-Z]+', ' ', regex=True)
    # collapse whitespace, lowercase, strip accents
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    s = s.apply(strip_accents_text)
    return s


def clean_title_series(s: pd.Series) -> pd.Series:
    s = clean_series_base(s)
    s = s.str.replace(NOISE_RE, ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def filename_to_titlelike(s: pd.Series) -> pd.Series:
    s = clean_series_base(s)
    # If pattern "Artist - Title", keep trailing part
    parts = s.str.split(' - ', n=1)
    s = parts.str[-1]
    s = clean_title_series(s)
    return s


def load_data(feat_path: Path, sy_path: Path):
    df_feat = pd.read_csv(feat_path, low_memory=False)
    df_sy   = pd.read_csv(sy_path,   low_memory=False)

    if "filename" not in df_feat.columns:
        raise ValueError("Expected 'filename' column in features CSV.")

    # Build normalized keys
    feat_key = filename_to_titlelike(df_feat["filename"])
    if "Title" in df_sy.columns:
        sy_key = clean_title_series(df_sy["Title"])
    else:
        sy_key = clean_title_series(df_sy.iloc[:, 0])

    left  = df_feat.copy()
    right = df_sy.copy()
    left["__k__"]  = feat_key
    right["__k__"] = sy_key
    return left, right


def exact_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(
        right, on="__k__", how="inner", suffixes=("_feat", "_sy")
    ).drop(columns=["__k__"])
    return merged


def pick_popularity_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def canonicalize_one_row_per_filename(df_merged_exact: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure one row per features filename.
    Prefer the row with highest popularity if a known popularity column exists,
    otherwise keep first occurrence.
    """
    pop_col = pick_popularity_column(df_merged_exact, POPULARITY_COLUMNS)
    if pop_col:
        df_sorted = df_merged_exact.sort_values(
            by=["filename", pop_col],
            ascending=[True, False],
            na_position="last"
        )
    else:
        df_sorted = df_merged_exact

    canonical = df_sorted.drop_duplicates(subset=["filename"], keep="first").reset_index(drop=True)
    return canonical


def build_blocks(strings: pd.Series):
    """Simple blocking index: first character -> list of indices."""
    blocks = {}
    for i, s in enumerate(strings):
        key = s[0] if (isinstance(s, str) and len(s) > 0) else ""
        blocks.setdefault(key, []).append(i)
    return blocks


def fuzzy_review(left: pd.DataFrame, right: pd.DataFrame,
                 exact_keys_left: set, threshold=90,
                 max_per_row=3, length_window=3, block_by_first_char=True) -> pd.DataFrame:
    """
    For unmatched left rows, propose top-N candidates from right using high-threshold fuzzy matching.
    """
    left_unmatched = left[~left["__k__"].isin(exact_keys_left)].copy()
    right_keys = right["__k__"].tolist()

    blocks = build_blocks(right["__k__"]) if block_by_first_char else None

    rows = []
    for _, row in left_unmatched.iterrows():
        key = row["__k__"]
        if not isinstance(key, str) or key == "":
            continue

        # preselect by first char + length window
        cand_indices = range(len(right_keys))
        if block_by_first_char:
            first = key[0] if len(key) > 0 else ""
            cand_indices = blocks.get(first, [])

        klen = len(key)
        cands = []
        for idx in cand_indices:
            rk = right_keys[idx]
            if not isinstance(rk, str) or rk == "":
                continue
            if abs(len(rk) - klen) <= length_window:
                cands.append((idx, rk))

        # score candidates
        scored = []
        for idx, rk in cands:
            score = fuzz.token_sort_ratio(key, rk)
            if score >= threshold:
                scored.append((score, idx, rk))

        if not scored:
            continue

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:max_per_row]

        for score, idx, rk in top:
            out = {
                "features_filename": row.get("filename", ""),
                "features_title_key": key,
                "sy_title_key": rk,
                "similarity": score
            }
            # include some helpful SyData columns for context if present
            for col in ["Title", "Artist", "Channel", "Track", "Track_URL_Spotify", "Spotify_Track_Popularity"]:
                if col in right.columns:
                    out[f"SyData_{col}"] = right.iloc[idx][col]
            rows.append(out)

    return pd.DataFrame(rows)


def auto_accept_fuzzy(exact_canonical: pd.DataFrame,
                      review_df: pd.DataFrame,
                      sy_df: pd.DataFrame,
                      threshold: int) -> pd.DataFrame:
    """
    Append very high-confidence fuzzy matches (>=threshold) for filenames not already matched.
    Joins fuzzy Title back to full SyData rows; enforces one row per filename at the end.
    """
    if review_df.empty:
        return exact_canonical.copy()

    fhi = review_df[review_df["similarity"] >= threshold].copy()
    if fhi.empty:
        return exact_canonical.copy()

    # one suggestion per filename (highest similarity)
    fhi = fhi.sort_values(["features_filename", "similarity"], ascending=[True, False])
    fhi = fhi.drop_duplicates(subset=["features_filename"], keep="first")

    # only filenames not already present
    existing = set(exact_canonical["filename"])
    fhi = fhi[~fhi["features_filename"].isin(existing)].copy()
    if fhi.empty:
        return exact_canonical.copy()

    # join fuzzy Title back to SyData for full columns
    if "Title" not in sy_df.columns:
        raise ValueError("SyData must have a 'Title' column for fuzzy auto-accept step.")
    fhi_joined = fhi.merge(
        sy_df,
        left_on="SyData_Title",
        right_on="Title",
        how="left",
        suffixes=("", "_sydup")
    )

    fhi_joined = fhi_joined.rename(columns={"features_filename": "filename"})

    # concat and re-enforce one row per filename
    combined = pd.concat([exact_canonical, fhi_joined], ignore_index=True, sort=False)
    combined = combined.sort_values(by=["filename"]).drop_duplicates(subset=["filename"], keep="first")
    return combined


def main():
    print("Loading data…")
    left, right = load_data(FEAT_PATH, SY_PATH)
    print(f"Features rows: {len(left)}; SyData rows: {len(right)}")

    print("Exact merge (strong normalization)…")
    merged_exact_raw = exact_merge(left, right)
    merged_exact_raw.to_csv(OUT_MERGED_EXACT_RAW, index=False)
    print(f"Exact merged (raw) rows: {len(merged_exact_raw)} → {OUT_MERGED_EXACT_RAW}")

    print("Canonicalizing to one row per filename…")
    exact_canonical = canonicalize_one_row_per_filename(merged_exact_raw)
    exact_canonical.to_csv(OUT_MERGED_EXACT_CANON, index=False)
    print(f"Exact merged (canonical) rows: {len(exact_canonical)} → {OUT_MERGED_EXACT_CANON}")

    if BUILD_FUZZY_REVIEW:
        print("Building fuzzy review candidates (high confidence)…")
        # Rebuild left keys (same normalization as load)
        # left['__k__'] already exists; exact key set comes from canonical exact set
        # compute canonical key from filename to line up with left['__k__']
        canon_keys = set(filename_to_titlelike(exact_canonical["filename"]))
        review = fuzzy_review(
            left=left,
            right=right,
            exact_keys_left=canon_keys,
            threshold=FUZZ_THRESHOLD,
            max_per_row=MAX_CANDIDATES_PER_ROW,
            length_window=LENGTH_WINDOW,
            block_by_first_char=BLOCK_BY_FIRST_CHAR,
        )
        review.to_csv(OUT_REVIEW_CANDS, index=False)
        print(f"Review candidates: {len(review)} → {OUT_REVIEW_CANDS}")
    else:
        review = pd.DataFrame()

    if AUTO_ACCEPT_FUZZY:
        print(f"Auto-accepting fuzzy ≥ {AUTO_ACCEPT_THRESHOLD} and appending…")
        final = auto_accept_fuzzy(
            exact_canonical=exact_canonical,
            review_df=review,
            sy_df=right.drop(columns=["__k__"]),
            threshold=AUTO_ACCEPT_THRESHOLD
        )
        final.to_csv(OUT_FINAL_WITH_FUZZY, index=False)
        print(f"Final (canonical + auto-fuzzy) rows: {len(final)} → {OUT_FINAL_WITH_FUZZY}")

    print("Done.")


if __name__ == "__main__":
    main()
