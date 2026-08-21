# this is what it looks like  now, chang it: 
#@app.callback(
 #   Output("tab-content", "children"),
 #   Input("tabs", "value")
#)
#def render_tab(tab):
 #   ...

#change to: 
# TEMP FIX FOR DASH INFINITE LOADING

from dash.exceptions import PreventUpdate

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    prevent_initial_call=True
)
def render_tab(tab):
    raise PreventUpdate
