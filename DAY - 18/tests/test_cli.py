import pandas as pd

from csv_predictor.cli import main


def _sample(tmp_path):
    path = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "age": [25, 52],
            "income": [40000.0, 90000.0],
            "score": [30.0, 80.0],
            "segment": ["A", "C"],
        }
    ).to_csv(path, index=False)
    return path


def test_cli_writes_predictions(tmp_path):
    out = tmp_path / "out.csv"
    assert main([str(_sample(tmp_path)), "-o", str(out)]) == 0
    result = pd.read_csv(out)
    assert len(result) == 2
    assert "prediction" in result.columns
    assert set(result["prediction"]).issubset({0, 1})


def test_cli_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"age": [1]}).to_csv(bad, index=False)
    assert main([str(bad), "-o", str(tmp_path / "o.csv")]) == 1


def test_cli_missing_input_file(tmp_path):
    assert main([str(tmp_path / "nope.csv"), "-o", str(tmp_path / "o.csv")]) == 1
