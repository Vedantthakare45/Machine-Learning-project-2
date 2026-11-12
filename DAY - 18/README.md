# csv-predictor

A scikit-learn pipeline (scaled numeric features + one-hot categoricals → gradient boosting) packaged as a Python library with a command-line tool.

```
predict-csv input.csv -o predictions.csv
```

## Project layout

```
csv-predictor/
├── setup.py                 # packaging config + console entry point
├── pyproject.toml           # build backend
├── MANIFEST.in
├── train_model.py           # trains the pipeline, saves csv_predictor/model.joblib
├── csv_predictor/
│   ├── __init__.py
│   ├── predict.py           # load pipeline, read CSV, write predictions
│   ├── cli.py               # argparse CLI (entry point: predict-csv)
│   └── model.joblib         # bundled trained pipeline
├── sample_data/input.csv    # example input
└── tests/test_cli.py
```

## 1. Set up and train

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas scikit-learn joblib pytest
python train_model.py        # regenerates csv_predictor/model.joblib
```

A trained `model.joblib` is already included, so you can skip training unless you change the pipeline.

## 2. Try it locally

```bash
pip install -e .
predict-csv sample_data/input.csv -o predictions.csv
pytest
```

The input CSV needs the columns `age`, `income`, `score`, `segment`. The output has the original columns plus `prediction` and `probability`.

## 3. Publish to TestPyPI and install via pip

1. Create accounts at https://test.pypi.org and generate an API token (Account settings → API tokens).
2. In `setup.py`, change `name` to something unique, for example `csv-predictor-<yourname>`.
3. Build and upload:

```bash
pip install build twine
python -m build
twine upload --repository testpypi dist/*
# username: __token__    password: <your pypi-... token>
```

4. Install from TestPyPI in a fresh virtualenv:

```bash
python -m venv /tmp/testenv && source /tmp/testenv/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            csv-predictor-<yourname>
predict-csv sample_data/input.csv -o out.csv
```

`--extra-index-url` is needed because TestPyPI does not host pandas, scikit-learn, etc.

## Using your own pipeline

Edit `build_pipeline()` and `make_data()` in `train_model.py` (or dump your own fitted pipeline to `csv_predictor/model.joblib`). If your pipeline uses custom transformer classes, put them inside the `csv_predictor` package so `joblib.load` can import them on the user's machine.

Note: a pickled model should be loaded with the same scikit-learn version it was trained with. Pin the version in `install_requires` for production use.
