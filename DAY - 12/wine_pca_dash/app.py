"""
Interactive PCA explorer for the Wine dataset (Dash + scikit-learn + Plotly).

Features
--------
1. Slider to choose n_components
2. Explained-variance bar chart (+ cumulative line) for the selected components
3. Scatter plot of PC1 vs PC2 that updates dynamically
"""

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
wine = load_wine(as_frame=True)
X_raw = wine.data
y = wine.target.map(dict(enumerate(wine.target_names)))  # class_0, class_1, class_2
N_FEATURES = X_raw.shape[1]  # 13

# PCA is scale-sensitive, so standardize first (mean 0, std 1 per feature)
X_scaled = StandardScaler().fit_transform(X_raw)

# ----------------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Wine PCA Explorer"

app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Inter, Arial, sans-serif",
    },
    children=[
        html.H1("Wine Dataset - PCA Explorer", style={"marginBottom": "4px"}),
        html.P(
            "Move the slider to change the number of principal components. "
            "The variance chart and the PC1 vs PC2 scatter update instantly.",
            style={"color": "#555"},
        ),
        # Step 1: slider for n_components
        html.Div(
            style={"margin": "24px 0 8px 0"},
            children=[
                html.Label(
                    "Number of components (n_components)",
                    style={"fontWeight": "600"},
                ),
                dcc.Slider(
                    id="n-components-slider",
                    min=2,  # need at least 2 for the PC1 vs PC2 scatter
                    max=N_FEATURES,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(2, N_FEATURES + 1)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        ),
        html.Div(id="summary-text", style={"margin": "16px 0", "fontWeight": "500"}),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                # Step 2: explained variance bar chart
                dcc.Graph(id="variance-graph", style={"flex": "1 1 480px"}),
                # Step 3: scatter of first two components
                dcc.Graph(id="scatter-graph", style={"flex": "1 1 480px"}),
            ],
        ),
    ],
)


# ----------------------------------------------------------------------------
# Callback
# ----------------------------------------------------------------------------
@app.callback(
    Output("variance-graph", "figure"),
    Output("scatter-graph", "figure"),
    Output("summary-text", "children"),
    Input("n-components-slider", "value"),
)
def update_graphs(n_components):
    # Fit PCA with the selected number of components
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_ * 100  # percent per component
    cum = evr.cumsum()
    labels = [f"PC{i + 1}" for i in range(n_components)]

    # ---- Explained variance bar chart (+ cumulative line) ----
    var_fig = go.Figure()
    var_fig.add_bar(
        x=labels,
        y=evr,
        name="Individual",
        marker_color="#6c5ce7",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    )
    var_fig.add_scatter(
        x=labels,
        y=cum,
        name="Cumulative",
        mode="lines+markers",
        line={"color": "#e17055"},
        hovertemplate="%{x} cumulative: %{y:.2f}%<extra></extra>",
    )
    var_fig.update_layout(
        title="Explained variance by component",
        xaxis_title="Principal component",
        yaxis_title="Explained variance (%)",
        yaxis_range=[0, 105],
        legend={"orientation": "h", "y": -0.25},
        template="plotly_white",
        margin={"t": 60, "b": 60},
    )

    # ---- Scatter of PC1 vs PC2 ----
    scatter_fig = px.scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        color=y.values,
        labels={
            "x": f"PC1 ({evr[0]:.1f}%)",
            "y": f"PC2 ({evr[1]:.1f}%)",
            "color": "Wine class",
        },
        title="First two principal components",
        template="plotly_white",
    )
    scatter_fig.update_traces(marker={"size": 8, "opacity": 0.8})
    scatter_fig.update_layout(margin={"t": 60, "b": 60})

    summary = (
        f"{n_components} components keep {cum[-1]:.2f}% of the total variance "
        f"(PC1 + PC2 = {cum[1]:.2f}%)."
    )
    return var_fig, scatter_fig, summary


if __name__ == "__main__":
    app.run(debug=True)
