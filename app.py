import pandas as pd
from dash import Dash, dcc, html

# --- Load your dataset ---
data = pd.read_csv("merged_final_data.csv")

# --- Force tempo and energy to be numeric ---
data["tempo"] = pd.to_numeric(data["tempo"], errors="coerce")
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# --- Drop missing values ---
clean_data = data.dropna(subset=["tempo", "energy"])

# --- Take a random sample of 200 rows ---
sample = clean_data.sample(min(200, len(clean_data)), random_state=42)

# --- Initialize the Dash app ---
app = Dash(__name__)

# --- Define Layout ---
app.layout = html.Div([
    html.H1("STA 160 Capstone Dashboard"),
    html.P("Exploring features from merged_final_data.csv"),

    dcc.Graph(
        id="scatter-tempo-energy",
        figure={
            "data": [
                {
                    "x": sample["tempo"],
                    "y": sample["energy"],
                    "mode": "markers",
                    "type": "scatter",
                    "marker": {"opacity": 0.7, "size": 9, "color": "blue"},
                },
            ],
            "layout": {
                "title": "Tempo vs Energy (200 random tracks)",
                "xaxis": {"title": "Tempo (BPM)", "type": "linear"},
                "yaxis": {"title": "Energy", "range": [0, 1]},
            }
        }
    )
])

# --- Run the server ---
if __name__ == "__main__":
    app.run(debug=True)
