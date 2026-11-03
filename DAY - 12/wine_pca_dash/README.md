# Wine PCA Explorer (Dash)

An interactive Dash app that applies PCA to the Wine dataset.

## What it does
1. **Slider** to choose `n_components` (2 to 13)
2. **Bar chart** of explained variance per component, with a cumulative line
3. **Scatter plot** of PC1 vs PC2, coloured by wine class, updating live

## Run it

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

## How it works
- Loads the Wine dataset from scikit-learn (178 samples, 13 features).
- Standardizes features with `StandardScaler` (PCA is scale-sensitive).
- On every slider change, fits `PCA(n_components=n)` and redraws both charts.
- Explained variance ratio = eigenvalue of each component / total variance.

## Files
- `app.py` - the full Dash app
- `requirements.txt` - Python dependencies
