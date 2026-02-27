# hour12_predictive_ml.py
"""
Hour 12 – Predictive ML Pipeline (Proactive vs. Reactive)
==========================================================
Trains a scikit-learn Random Forest Regressor on historical village data
to predict *tomorrow's* outbreak severity score, enabling the OR-Tools
solver to route vans proactively rather than reactively.

Pipeline
--------
  generate_synthetic_history()   → creates training dataset if none exists
  train_severity_model()         → fits & evaluates the Random Forest
  predict_tomorrow()             → returns severity scores for all villages
  enrich_data_with_predictions() → patches data['cases'] used by the solver
  save_model() / load_model()    → joblib persistence for the Streamlit app

Streamlit Integration (see bottom of file)
------------------------------------------
  Copy the `streamlit_sidebar_snippet()` docstring into your app.py sidebar
  section to add the "Predictive Mode" toggle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/severity_rf_model.joblib")
HISTORY_PATH = Path("data/village_history.csv")

# ── Feature & target columns ──────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "Population",
    "Baseline_Cases",
    "Water_Quality_Index",   # 0–100, lower = worse
    "Distance_to_PHC_km",   # distance to nearest Primary Health Centre
    "Avg_Temp_C",
    "Avg_Humidity_Pct",
    "Prev_Severity_T",       # today's severity (lag-1 feature)
]
TARGET_COL: str = "Severity_T_plus_1"


# ── 1. Synthetic data generator ───────────────────────────────────────────────

def generate_synthetic_history(
    n_villages: int = 20,
    n_weeks: int = 26,
    random_state: int = 42,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a plausible synthetic historical dataset for training.

    In production this would be replaced by the State Health Department's
    weekly surveillance feed (IHIP / IDSP portal data).
    """
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []

    for v in range(n_villages):
        vid = f"V{v+1:02d}"
        population = int(rng.integers(800, 8_000))
        baseline = int(population * rng.uniform(0.005, 0.04))
        wqi = round(float(rng.uniform(20.0, 90.0)), 1)
        dist_phc = round(float(rng.uniform(2.0, 45.0)), 1)

        prev_severity = round(float(rng.uniform(1.0, 5.0)), 2)

        for week in range(n_weeks):
            temp = round(float(rng.normal(32.0, 5.0)), 1)
            humidity = int(rng.integers(40, 95))

            # Ground-truth severity: domain-driven formula + noise
            severity_tomorrow = (
                0.0008 * population
                + 0.30 * baseline
                - 0.04 * wqi          # better water quality → lower severity
                + 0.15 * dist_phc     # farther from PHC → worse outcomes
                + 0.12 * temp
                + 0.02 * humidity
                + 0.60 * prev_severity
                + rng.normal(0, 0.5)
            )
            severity_tomorrow = float(np.clip(severity_tomorrow, 0.0, 10.0))

            records.append({
                "Village_ID":           vid,
                "Week":                 week,
                "Population":           population,
                "Baseline_Cases":       baseline,
                "Water_Quality_Index":  wqi,
                "Distance_to_PHC_km":  dist_phc,
                "Avg_Temp_C":          temp,
                "Avg_Humidity_Pct":    humidity,
                "Prev_Severity_T":     round(prev_severity, 2),
                TARGET_COL:            round(severity_tomorrow, 2),
            })
            prev_severity = severity_tomorrow   # roll forward

    df = pd.DataFrame(records)

    if save:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(HISTORY_PATH, index=False)
        log.info("Synthetic history saved → %s  (%d rows)", HISTORY_PATH, len(df))

    return df


# ── 2. Model training ─────────────────────────────────────────────────────────

def train_severity_model(
    df: pd.DataFrame | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """
    Train a Random Forest Regressor wrapped in a sklearn Pipeline.

    Returns
    -------
    pipeline : Fitted sklearn Pipeline (StandardScaler → RandomForest)
    metrics  : dict with MAE and R² on the held-out test set
    """
    if df is None:
        if HISTORY_PATH.exists():
            df = pd.read_csv(HISTORY_PATH)
            log.info("Loaded history from %s  (%d rows)", HISTORY_PATH, len(df))
        else:
            log.warning("No history file found – generating synthetic data …")
            df = generate_synthetic_history()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics: dict[str, float] = {
        "MAE":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "R2":   round(float(r2_score(y_test, y_pred)), 4),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }

    log.info(
        "Model trained │ MAE=%.4f │ R²=%.4f │ train=%d test=%d",
        metrics["MAE"], metrics["R2"], metrics["n_train"], metrics["n_test"],
    )

    # Feature importances (useful for Streamlit explainability panel)
    rf_model = pipeline.named_steps["rf"]
    importances = dict(zip(FEATURE_COLS, rf_model.feature_importances_.round(4)))
    log.info("Feature importances: %s", importances)

    return pipeline, metrics


# ── 3. Persistence ────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Serialize fitted pipeline to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    log.info("Model saved → %s", path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a previously serialized pipeline.  Trains a fresh one if not found."""
    if not path.exists():
        log.warning("Model not found at %s – training fresh model …", path)
        pipeline, _ = train_severity_model()
        save_model(pipeline, path)
        return pipeline
    pipeline: Pipeline = joblib.load(path)
    log.info("Model loaded from %s", path)
    return pipeline


# ── 4. Inference ──────────────────────────────────────────────────────────────

def predict_tomorrow(
    pipeline: Pipeline,
    current_village_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict tomorrow's severity score for every village in `current_village_df`.

    The dataframe must contain (or will receive defaults for) all FEATURE_COLS.
    Missing columns are filled with sensible defaults so the function degrades
    gracefully when only partial data is available.

    Returns
    -------
    df : Input dataframe with an added `Predicted_Severity_T_plus_1` column,
         and a derived `Predicted_Active_Cases` column scaled from Population.
    """
    df = current_village_df.copy()

    # ── Apply defaults for any missing feature columns ─────────────────────
    defaults: dict[str, float] = {
        "Population":          3_000.0,
        "Baseline_Cases":        30.0,
        "Water_Quality_Index":   55.0,
        "Distance_to_PHC_km":   15.0,
        "Avg_Temp_C":            32.0,
        "Avg_Humidity_Pct":      65.0,
        "Prev_Severity_T":        3.0,
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
            log.debug("Column '%s' missing – using default %.1f", col, default_val)

    X = df[FEATURE_COLS].fillna(0).values
    predictions: np.ndarray = pipeline.predict(X)
    predictions = np.clip(predictions, 0.0, 10.0)

    df["Predicted_Severity_T_plus_1"] = predictions.round(2)

    # Scale severity (0–10) to approximate active case counts for the solver
    # Rule: severity 5.0 on a village of 3000 ≈ 30 cases
    df["Predicted_Active_Cases"] = (
        (df["Predicted_Severity_T_plus_1"] / 10.0)
        * df["Population"]
        * 0.02
    ).round(0).astype(int)

    return df


def enrich_data_with_predictions(
    data: dict[str, Any],
    pipeline: Pipeline,
    current_village_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Replace `data['cases']` with tomorrow's predicted active cases.
    This is the single line that flips the solver from Reactive → Proactive.

    Parameters
    ----------
    data               : OR-Tools data dict (from build_data_model).
    pipeline           : Fitted sklearn Pipeline.
    current_village_df : DataFrame with village features (same row order as data).

    Returns
    -------
    data : Mutated dict with updated `cases` and new `predicted_df` key.
    """
    predicted_df = predict_tomorrow(pipeline, current_village_df)
    predicted_cases: list[int] = predicted_df["Predicted_Active_Cases"].tolist()

    log.info(
        "Predictive mode: replacing today's cases with tomorrow's predictions. "
        "Top 3 hotspots: %s",
        predicted_df.nlargest(3, "Predicted_Severity_T_plus_1")[
            ["Village_ID", "Predicted_Severity_T_plus_1", "Predicted_Active_Cases"]
        ].to_dict("records"),
    )

    data["cases"] = predicted_cases
    data["predicted_df"] = predicted_df   # pass through for UI display
    return data


# ── 5. Streamlit sidebar snippet (copy into app.py) ──────────────────────────

STREAMLIT_SIDEBAR_SNIPPET = '''
# ── Paste this block into your app.py sidebar section ────────────────────────

import streamlit as st
from hour12_predictive_ml import load_model, enrich_data_with_predictions

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("🔮 Routing Intelligence")

    predictive_mode: bool = st.toggle(
        "Predictive Mode (Tomorrow's Hotspots)",
        value=False,
        help=(
            "When ON, the solver routes vans based on ML-predicted severity "
            "scores for tomorrow instead of today's active case counts. "
            "Powered by a Random Forest trained on historical surveillance data."
        ),
    )

    if predictive_mode:
        st.info(
            "🧠 **Proactive routing enabled.**  "
            "The solver is optimising for predicted outbreak locations.",
            icon="🔮",
        )

# ── In your main solve block ──────────────────────────────────────────────────
# (Assuming `data` is already built by build_data_model and
#  `village_df` is your merged village DataFrame)

if predictive_mode:
    try:
        ml_pipeline = load_model()               # loads or trains automatically
        data = enrich_data_with_predictions(
            data=data,
            pipeline=ml_pipeline,
            current_village_df=village_df,
        )

        # Show predicted hotspot table in the UI
        pred_df = data.get("predicted_df")
        if pred_df is not None:
            st.subheader("📊 Predicted Severity – Tomorrow")
            st.dataframe(
                pred_df[["Village_ID",
                          "Predicted_Severity_T_plus_1",
                          "Predicted_Active_Cases"]]
                  .sort_values("Predicted_Severity_T_plus_1", ascending=False)
                  .reset_index(drop=True),
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Predictive mode error: {e}")
        st.warning("Falling back to today's active case counts.")
# ─────────────────────────────────────────────────────────────────────────────
'''


def print_streamlit_snippet() -> None:
    """Print the Streamlit integration snippet for easy copy-paste."""
    print(STREAMLIT_SIDEBAR_SNIPPET)


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Hour 12 – Predictive ML Pipeline  (smoke test)")
    print("=" * 60)

    # 1. Generate / train
    df_history = generate_synthetic_history(n_villages=20, n_weeks=26)
    pipeline, metrics = train_severity_model(df_history)
    save_model(pipeline)

    print(f"\n✅  Training complete │ MAE={metrics['MAE']} │ R²={metrics['R2']}")

    # 2. Simulate current village snapshot
    current = pd.DataFrame({
        "Village_ID":         [f"V{i:02d}" for i in range(1, 6)],
        "Population":         [1200, 4500, 800, 3200, 5600],
        "Baseline_Cases":     [12, 45, 8, 32, 56],
        "Water_Quality_Index":[75, 30, 85, 50, 20],
        "Distance_to_PHC_km": [5, 25, 8, 18, 40],
        "Avg_Temp_C":         [34, 33, 31, 35, 36],
        "Avg_Humidity_Pct":   [70, 85, 60, 75, 90],
        "Prev_Severity_T":    [2.5, 6.1, 1.8, 4.3, 7.2],
    })

    # 3. Predict
    result_df = predict_tomorrow(pipeline, current)
    print("\n── Tomorrow's Predictions ───────────────────────────")
    print(
        result_df[["Village_ID",
                    "Predicted_Severity_T_plus_1",
                    "Predicted_Active_Cases"]].to_string(index=False)
    )

    print("\n── Streamlit Sidebar Snippet ────────────────────────")
    print_streamlit_snippet()
