from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "usgs_earthquakes_all_month.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    clean = df.dropna(subset=["magnitude", "latitude", "longitude", "depth_km", "time"]).copy()
    clean = clean[clean["type"].fillna("earthquake").eq("earthquake")].copy()

    clean["hour"] = clean["time"].dt.hour
    clean["weekday"] = clean["time"].dt.day_name()
    clean["day_of_month"] = clean["time"].dt.day
    clean["abs_latitude"] = clean["latitude"].abs()
    clean["log_depth_km"] = np.log1p(clean["depth_km"].clip(lower=0))
    clean["hemisphere"] = np.where(clean["latitude"] >= 0, "Northern", "Southern")
    clean["east_west"] = np.where(clean["longitude"] >= 0, "Eastern", "Western")
    clean["geo_quadrant"] = clean["hemisphere"] + " / " + clean["east_west"]
    clean["felt"] = clean["felt"].fillna(0)
    clean["tsunami"] = clean["tsunami"].fillna(0).astype(int)
    clean["status"] = clean["status"].fillna("unknown")
    clean["magnitude_type"] = clean["magnitude_type"].fillna("unknown")
    clean["net"] = clean["net"].fillna("unknown")
    clean["is_strong_event"] = (clean["magnitude"] >= 4.5).astype(int)
    return clean


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "depth_km",
    "abs_latitude",
    "log_depth_km",
    "hour",
    "day_of_month",
    "felt",
    "tsunami",
    "magnitude_type",
    "status",
    "net",
    "geo_quadrant",
]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in FEATURE_COLUMNS if col not in numeric_features]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def predict_score(model: Pipeline, X_values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_values)[:, 1]
    raw = model.decision_function(X_values)
    return 1 / (1 + np.exp(-raw))


def evaluate(model: Pipeline, X_values: pd.DataFrame, y_true: pd.Series, threshold: float = 0.5) -> dict:
    score = predict_score(model, X_values)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "average_precision": average_precision_score(y_true, score),
        "brier_score": brier_score_loss(y_true, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    clean = load_clean_data()
    X = clean[FEATURE_COLUMNS].copy()
    y = clean["is_strong_event"].copy()
    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=18,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            random_state=RANDOM_STATE,
        ),
    }

    random_rows = []
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    for name, estimator in models.items():
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        model.fit(X_train, y_train)
        random_rows.append({"model": name, "validation": "random_stratified_75_25", **evaluate(model, X_test, y_test)})

    spatial_rows = []
    for holdout_region in sorted(clean["geo_quadrant"].unique()):
        train_mask = clean["geo_quadrant"] != holdout_region
        test_mask = clean["geo_quadrant"] == holdout_region
        X_region_train, y_region_train = X.loc[train_mask], y.loc[train_mask]
        X_region_test, y_region_test = X.loc[test_mask], y.loc[test_mask]
        for name, estimator in models.items():
            model = Pipeline([("preprocess", build_preprocessor(X_region_train)), ("model", estimator)])
            model.fit(X_region_train, y_region_train)
            spatial_rows.append(
                {
                    "model": name,
                    "validation": "leave_one_geographic_quadrant_out",
                    "holdout_region": holdout_region,
                    **evaluate(model, X_region_test, y_region_test),
                }
            )

    spatial_df = pd.DataFrame(spatial_rows)
    random_df = pd.DataFrame(random_rows)
    spatial_summary = (
        spatial_df.groupby("model", as_index=False)
        .agg(
            holdout_regions=("holdout_region", "count"),
            total_holdout_n=("n", "sum"),
            total_holdout_positives=("positives", "sum"),
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_recall=("recall", "mean"),
            min_recall=("recall", "min"),
            mean_precision=("precision", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .sort_values(["mean_roc_auc", "mean_recall"], ascending=False)
    )

    random_df.to_csv(OUTPUT_DIR / "random_split_extended_metrics.csv", index=False)
    spatial_df.to_csv(OUTPUT_DIR / "spatial_holdout_by_region.csv", index=False)
    spatial_summary.to_csv(OUTPUT_DIR / "spatial_holdout_summary.csv", index=False)

    manifest = {
        "records_used": int(len(clean)),
        "time_range_utc": [clean["time"].min().isoformat(), clean["time"].max().isoformat()],
        "spatial_validation": "Leave-one-geographic-quadrant-out validation. Each quadrant is held out once while the model is trained on the other three quadrants.",
        "feature_columns": FEATURE_COLUMNS,
        "leakage_controls": ["magnitude excluded", "USGS significance excluded", "title/place/url excluded"],
        "outputs": [
            "outputs/random_split_extended_metrics.csv",
            "outputs/spatial_holdout_by_region.csv",
            "outputs/spatial_holdout_summary.csv",
        ],
    }
    (OUTPUT_DIR / "spatial_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Random split extended metrics:")
    print(random_df.round(3).to_string(index=False))
    print("\nSpatial holdout by region:")
    print(spatial_df.round(3).to_string(index=False))
    print("\nSpatial holdout summary:")
    print(spatial_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
