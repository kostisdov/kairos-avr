# Notebooks — Proof-of-Concept Implementation

> This folder is **optional** but evaluated positively if present.

Place your Jupyter notebooks here. A strong submission includes:

- Data loading and exploratory analysis
- Feature engineering pipeline
- Model training and evaluation
- SHAP / feature importance visualisation

## Suggested Notebook Structure

```
notebooks/
├── 01_eda.ipynb                 # Exploratory data analysis on your chosen dataset
├── 02_feature_engineering.ipynb # Feature construction (gradient progression rate, EOA index, PPM flag, etc.)
├── 03_model_training.ipynb      # Model training, cross-validation, hyperparameter tuning
└── 04_evaluation.ipynb          # Metrics, calibration, SHAP plots, subgroup analysis
```

You can combine these into a single notebook if preferred — the split is just for readability.

## Environment

Document your dependencies here so the panel can reproduce your results:

```
python >= 3.10
pandas
numpy
scikit-learn
xgboost       # or your chosen framework
lifelines     # or scikit-survival, for time-to-event modelling
shap
matplotlib
jupyter
```

Or include a `requirements.txt` / `environment.yml` in this folder.
