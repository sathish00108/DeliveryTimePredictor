# Agent Instructions

## Project Overview
This is a **Delivery Time Regression** machine-learning project.

- **Dataset:** `data/dataset.csv` — 43 000+ e-commerce delivery records
- **Target:** `Delivery_Time` (minutes, continuous) → **Regression**
- **Best Model:** Selected automatically from Ridge, Random Forest, Extra Trees, Gradient Boosting

## Key Design Decisions
1. **Feature Engineering** — Haversine distance, prep time, day-of-week, and hour extracted from raw coordinates and timestamps.
2. **Pipeline** — A single `sklearn.pipeline.Pipeline` encapsulates preprocessing (imputation + scaling for numerics, imputation + ordinal encoding for categoricals) and the estimator. This ensures no data leakage and seamless serialisation.
3. **Model Selection** — All models are evaluated on a 20 % hold-out set; the one with the lowest MAE is saved.
4. **Serialisation** — The entire pipeline _plus_ metadata (unique category values, numeric stats, metric results) is saved as `models/model.pkl` using `joblib`. `app.py` loads this single file and reads category lists from it dynamically — no hard-coding.

## How to Retrain
```bash
python train_model.py
```
The script will overwrite `models/model.pkl` with the newly trained pipeline.

## How to Run the App
```bash
streamlit run app.py
```

## Adding a New Model
In `train_model.py`, add an entry to the `candidates` dict:
```python
candidates["My Model"] = MyRegressor(...)
```
Retrain and the best model will be automatically selected.

## No API Keys Required
This project runs fully offline. See `.env.example` for the (empty) environment variable template.
