from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="csv-predictor-yourname",  # must be unique on TestPyPI: change "yourname"
    version="0.1.0",
    description="Packaged scikit-learn pipeline with a CLI that predicts from a CSV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"csv_predictor": ["model.joblib"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
    entry_points={
        "console_scripts": [
            "predict-csv=csv_predictor.cli:main",
        ],
    },
)
