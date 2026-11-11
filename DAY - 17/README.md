# Streamlit SHAP Explainer Dashboard

Interactive SHAP explanations for RandomForest predictions on user-selected test-set rows.

## Project layout
```
train_model.py     # trains RandomForest, builds SHAP explainer, saves artifacts/
app.py             # Streamlit dashboard
requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python train_model.py           # creates artifacts/
streamlit run app.py
```

## What the dashboard does
1. Loads the trained RandomForest and SHAP `TreeExplainer` from `artifacts/`.
2. Sidebar lets you pick any row from the held-out test set and shows predicted vs. actual class.
3. Displays a SHAP force plot for the selected instance, a bar chart of its top feature contributions
   (red pushes toward the explained class, blue pushes away), and the model's global feature importance.

To use your own model/data, replace the dataset in `train_model.py` and keep the same artifact names.
