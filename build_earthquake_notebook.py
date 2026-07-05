import json
from pathlib import Path
from textwrap import dedent


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in dedent(text).strip().splitlines()],
    }


cells = [
    md(
        """
        # Global Earthquake Activity Analysis and Strong-Event Prediction

        This notebook is my end-to-end data science project using a recent global earthquake snapshot from the United States Geological Survey (USGS). The aim is to explore where earthquakes are happening, how magnitude and depth behave, and whether a machine learning model can identify stronger earthquake events from available event features.

        Dataset source: [USGS Earthquake Hazards Program - all earthquakes, past month](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson)

        The workflow includes data cleaning, feature engineering, visual analysis, model comparison, hyperparameter tuning, and saving the best model.
        """
    ),
    md(
        """
        ## 1. Imports and Project Setup

        I keep the setup compact and reproducible. The notebook reads from the saved CSV snapshot in the `data` folder so the analysis can be rerun even if the live API changes later.
        """
    ),
    code(
        """
        import json
        import warnings
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", palette="viridis")

        DATA_PATH = Path("data/usgs_earthquakes_all_month.csv")
        OUTPUT_DIR = Path("outputs")
        MODEL_DIR = Path("models")
        OUTPUT_DIR.mkdir(exist_ok=True)
        MODEL_DIR.mkdir(exist_ok=True)

        RANDOM_STATE = 42
        """
    ),
    md(
        """
        ## 2. Load and Understand the Data

        The dataset contains earthquakes reported by USGS over the past month. Each row is one event, with location, magnitude, depth, reporting network, event status, and other USGS event metadata.
        """
    ),
    code(
        """
        df = pd.read_csv(DATA_PATH)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["updated"] = pd.to_datetime(df["updated"], utc=True, errors="coerce")

        print("Rows and columns:", df.shape)
        display(df.head())
        display(df.info())
        display(df.describe(include="all").T)
        """
    ),
    md(
        """
        ## 3. Cleaning and Feature Engineering

        I remove rows without a valid magnitude or location, then create features that are easy to explain:

        - hour, weekday, and day of month
        - absolute latitude, because distance from the equator can matter geologically
        - log-transformed depth, because depth has a long-tailed distribution
        - broad geographic quadrant
        - a binary target called `is_strong_event`, where magnitude 4.5 and above is treated as stronger
        """
    ),
    code(
        """
        clean = df.copy()
        clean = clean.dropna(subset=["magnitude", "latitude", "longitude", "depth_km", "time"])
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

        print("Clean rows:", clean.shape[0])
        print("Strong event share:")
        print(clean["is_strong_event"].value_counts(normalize=True).rename("proportion"))
        display(clean.head())
        """
    ),
    md(
        """
        ## 4. Earthquake Activity Overview

        These first charts show the main shape of the data: most events are small, depth is unevenly distributed, and stronger events are a smaller but important part of the dataset.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.histplot(clean["magnitude"], bins=40, kde=True, ax=axes[0])
        axes[0].axvline(4.5, color="red", linestyle="--", label="Strong threshold")
        axes[0].set_title("Magnitude distribution")
        axes[0].legend()

        sns.histplot(clean["depth_km"], bins=50, ax=axes[1])
        axes[1].set_title("Depth distribution")
        axes[1].set_xlabel("Depth (km)")

        strong_counts = clean["is_strong_event"].map({0: "Lower magnitude", 1: "Strong event"}).value_counts()
        sns.barplot(x=strong_counts.index, y=strong_counts.values, ax=axes[2])
        axes[2].set_title("Strong-event class balance")
        axes[2].set_ylabel("Number of events")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 5. Global Location Patterns

        This scatter map-style chart uses longitude and latitude to show where the recent earthquakes happened. Larger and warmer points represent stronger magnitudes.
        """
    ),
    code(
        """
        plt.figure(figsize=(14, 7))
        scatter = plt.scatter(
            clean["longitude"],
            clean["latitude"],
            c=clean["magnitude"],
            s=np.clip(clean["magnitude"] ** 2, 5, 80),
            cmap="magma",
            alpha=0.65,
        )
        plt.colorbar(scatter, label="Magnitude")
        plt.title("Global earthquake locations by magnitude")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 6. Time and Region Patterns

        I look at activity by weekday, hour, and broad geographic quadrant. This is useful for storytelling and for checking whether the engineered features may carry signal.
        """
    ),
    code(
        """
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        sns.countplot(data=clean, x="weekday", order=weekday_order, ax=axes[0, 0])
        axes[0, 0].set_title("Earthquakes by weekday")
        axes[0, 0].tick_params(axis="x", rotation=35)

        sns.countplot(data=clean, x="hour", ax=axes[0, 1])
        axes[0, 1].set_title("Earthquakes by hour (UTC)")

        quadrant_rate = clean.groupby("geo_quadrant")["is_strong_event"].mean().sort_values(ascending=False)
        sns.barplot(x=quadrant_rate.values, y=quadrant_rate.index, ax=axes[1, 0])
        axes[1, 0].set_title("Strong-event rate by broad geographic quadrant")
        axes[1, 0].set_xlabel("Share of events >= magnitude 4.5")

        top_regions = clean["net"].value_counts().head(10)
        sns.barplot(x=top_regions.values, y=top_regions.index, ax=axes[1, 1])
        axes[1, 1].set_title("Top reporting networks")
        axes[1, 1].set_xlabel("Event count")

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 7. Relationships Between Magnitude, Depth, and Signal

        Magnitude, depth, felt reports, and significance are not independent. I show `sig` in the analysis because it is useful context, but I leave it out of the model because USGS significance is partly derived from magnitude and would leak the answer.
        """
    ),
    code(
        """
        numeric_view = clean[["magnitude", "depth_km", "felt", "sig", "latitude", "longitude", "abs_latitude", "hour", "tsunami", "is_strong_event"]]

        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_view.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation heatmap")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=clean.sample(min(4000, len(clean)), random_state=RANDOM_STATE),
            x="depth_km",
            y="magnitude",
            hue="is_strong_event",
            alpha=0.55,
        )
        plt.title("Magnitude vs depth")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 8. Prepare Machine Learning Data

        The target is `is_strong_event`. I avoid using the raw title, URL, exact place text, raw magnitude, and USGS significance score because those would either be unstable text fields or leak the answer. The model uses location, time, depth, and compact event metadata.
        """
    ),
    code(
        """
        feature_columns = [
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

        target = "is_strong_event"

        X = clean[feature_columns].copy()
        y = clean[target].copy()

        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = [col for col in feature_columns if col not in numeric_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ]
        )

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)
        print("Numeric features:", numeric_features)
        print("Categorical features:", categorical_features)
        """
    ),
    md(
        """
        ## 9. Train and Compare Baseline Models

        I compare four models: Logistic Regression, Random Forest, Extra Trees, and a tuned-style Gradient Boosting alternative from scikit-learn. The goal is to see which family handles this dataset best before tuning.
        """
    ),
    code(
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=400, max_depth=18, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=RANDOM_STATE),
        }

        fitted_models = {}
        rows = []

        def predict_score(model, X_values):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_values)[:, 1]
            raw = model.decision_function(X_values)
            return 1 / (1 + np.exp(-raw))

        def evaluate(model, X_values, y_true):
            score = predict_score(model, X_values)
            pred = (score >= 0.5).astype(int)
            return {
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, score),
                "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
            }

        for name, estimator in models.items():
            print(f"Training {name}...")
            pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            fitted_models[name] = pipe
            rows.append({"model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}, "confusion_matrix": metrics["confusion_matrix"]})
            print(metrics)

        baseline_results = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
        display(baseline_results)
        baseline_results.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False)
        """
    ),
    md(
        """
        ## 10. Visual Model Comparison

        This chart makes it easier to compare model tradeoffs. For this project, ROC-AUC is the main ranking metric because the stronger-event class is smaller than the lower-magnitude class.
        """
    ),
    code(
        """
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        ax = baseline_results.set_index("model")[metric_cols].sort_values("roc_auc").plot(kind="barh", figsize=(12, 7))
        ax.set_title("Baseline model comparison")
        ax.set_xlabel("Score")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        best_baseline_name = baseline_results.iloc[0]["model"]
        best_baseline = fitted_models[best_baseline_name]

        cm = np.array(baseline_results.iloc[0]["confusion_matrix"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion matrix: {best_baseline_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0.5, 1.5], ["Lower", "Strong"])
        plt.yticks([0.5, 1.5], ["Lower", "Strong"], rotation=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 11. Tune the Best Model Family

        I tune the best-performing baseline model with randomized search. This keeps the search practical while still exploring a useful range of settings.
        """
    ),
    code(
        """
        best_name = baseline_results.iloc[0]["model"]
        print("Best baseline model:", best_name)

        if best_name == "Extra Trees":
            estimator = ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
            param_space = {
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [10, 16, 22, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", None],
            }
        elif best_name == "Random Forest":
            estimator = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
            param_space = {
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [10, 16, 22, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", None],
            }
        elif best_name == "Hist Gradient Boosting":
            estimator = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            param_space = {
                "model__max_iter": [150, 250, 350],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            }
        else:
            estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
            param_space = {"model__C": np.logspace(-2, 2, 9)}

        tuning_pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        search = RandomizedSearchCV(
            tuning_pipe,
            param_distributions=param_space,
            n_iter=min(12, np.prod([len(v) for v in param_space.values()])),
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        search.fit(X_train, y_train)
        tuned_model = search.best_estimator_
        tuned_metrics = evaluate(tuned_model, X_test, y_test)

        print("Best parameters:")
        print(search.best_params_)
        print("Tuned metrics:")
        print(tuned_metrics)

        tuned_results = pd.DataFrame([{"model": best_name, "best_cv_roc_auc": search.best_score_, "best_params": search.best_params_, **{k: v for k, v in tuned_metrics.items() if k != "confusion_matrix"}, "confusion_matrix": tuned_metrics["confusion_matrix"]}])
        display(tuned_results)
        tuned_results.to_csv(OUTPUT_DIR / "tuned_model_results.csv", index=False)
        """
    ),
    md(
        """
        ## 12. Feature Importance

        If the final model exposes feature importance or coefficients, I use that to understand what drove the predictions. This is not causal proof, but it gives a useful model-level explanation.
        """
    ),
    code(
        """
        final_model = tuned_model
        model_step = final_model.named_steps["model"]
        preprocessor_step = final_model.named_steps["preprocess"]

        feature_names = preprocessor_step.get_feature_names_out()

        if hasattr(model_step, "feature_importances_"):
            importance_df = pd.DataFrame({"feature": feature_names, "importance": model_step.feature_importances_})
            importance_df["absolute_importance"] = importance_df["importance"].abs()
            title = "Top model features"
        elif hasattr(model_step, "coef_"):
            coefficients = model_step.coef_.ravel()
            importance_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
            importance_df["absolute_importance"] = importance_df["coefficient"].abs()
            title = "Top model coefficients by absolute size"
        else:
            importance_df = pd.DataFrame()
            print("Feature importance is not available for this model.")

        if not importance_df.empty:
            importance_df = importance_df.sort_values("absolute_importance", ascending=False).head(20)
            display(importance_df)
            importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

            plt.figure(figsize=(10, 7))
            value_col = "importance" if "importance" in importance_df.columns else "coefficient"
            sns.barplot(data=importance_df, x=value_col, y="feature")
            plt.title(title)
            plt.tight_layout()
            plt.show()
        """
    ),
    md(
        """
        ## 13. Save the Final Model

        The final tuned pipeline is saved with preprocessing included, so it can accept the original feature columns at prediction time.
        """
    ),
    code(
        """
        model_path = MODEL_DIR / "best_earthquake_strength_model.joblib"
        metadata_path = MODEL_DIR / "best_earthquake_strength_model_metadata.json"

        joblib.dump(final_model, model_path)

        metadata = {
            "dataset_source": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson",
            "target": target,
            "positive_class": "magnitude >= 4.5",
            "feature_columns": feature_columns,
            "best_model": best_name,
            "best_params": search.best_params_,
            "metrics": tuned_metrics,
            "records_used": int(len(clean)),
        }

        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

        print(f"Saved model to: {model_path.resolve()}")
        print(f"Saved metadata to: {metadata_path.resolve()}")
        print(json.dumps(metadata, indent=2, default=str))
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("earthquake_activity_analysis.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Created earthquake_activity_analysis.ipynb")
