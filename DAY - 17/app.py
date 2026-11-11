"""Streamlit dashboard: interactive SHAP explanations for user-selected predictions."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="SHAP Explainer Dashboard", layout="wide")


# ---- Step 1: load trained RandomForest and SHAP explainer ----
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS / "rf_model.joblib")
    explainer = joblib.load(ARTIFACTS / "shap_explainer.joblib")
    class_names = joblib.load(ARTIFACTS / "class_names.joblib")
    test_df = pd.read_csv(ARTIFACTS / "test_set.csv")
    return model, explainer, class_names, test_df


def class_slice(values, expected, cls):
    """Handle both old (list per class) and new (n, features, classes) SHAP output."""
    if isinstance(values, list):
        return np.asarray(values[cls]), float(np.atleast_1d(expected)[cls])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, cls], float(np.atleast_1d(expected)[cls])
    return values, float(np.atleast_1d(expected)[0])


if not (ARTIFACTS / "rf_model.joblib").exists():
    st.error("Artifacts not found. Run `python train_model.py` first.")
    st.stop()

model, explainer, class_names, test_df = load_artifacts()
X_test = test_df.drop(columns="target")
y_test = test_df["target"]

st.title("SHAP Explanations for RandomForest Predictions")

# ---- Step 2: UI to select a data point from the test set ----
st.sidebar.header("Select a data point")
idx = st.sidebar.selectbox("Test-set row index", options=list(X_test.index), index=0)
instance = X_test.loc[[idx]]

proba = model.predict_proba(instance)[0]
pred = int(np.argmax(proba))
st.sidebar.metric("Predicted class", class_names[pred], f"{proba[pred]:.1%} confidence")
st.sidebar.write(f"**Actual class:** {class_names[int(y_test.loc[idx])]}")

explain_cls = st.sidebar.radio(
    "Explain class",
    options=list(range(len(class_names))),
    index=pred,
    format_func=lambda i: class_names[i],
)

# Compute SHAP values for the selected instance
raw_values = explainer.shap_values(instance)
sv, base = class_slice(raw_values, explainer.expected_value, explain_cls)
sv = sv[0]

# ---- Step 3: force plot + feature importance for the selected instance ----
st.subheader(f"Force plot — explaining '{class_names[explain_cls]}'")
shap.force_plot(base, sv, instance.iloc[0], matplotlib=True, show=False)
st.pyplot(plt.gcf(), clear_figure=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature contributions (this instance)")
    contrib = (
        pd.DataFrame({"feature": X_test.columns, "shap_value": sv})
        .assign(abs_value=lambda d: d.shap_value.abs())
        .sort_values("abs_value", ascending=False)
        .head(15)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib.shap_value]
    ax.barh(contrib.feature, contrib.shap_value, color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)")
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Global feature importance (model)")
    imp = (
        pd.Series(model.feature_importances_, index=X_test.columns)
        .sort_values(ascending=False)
        .head(15)
        .iloc[::-1]
    )
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.barh(imp.index, imp.values, color="#2ca02c")
    ax2.set_xlabel("Importance")
    st.pyplot(fig2, clear_figure=True)

with st.expander("Selected instance feature values"):
    st.dataframe(instance.T.rename(columns={idx: "value"}))
