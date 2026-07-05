# Model Card: Earthquake Strong-Event Classifier

## Model Overview

This model classifies whether a reported earthquake event is a stronger event, defined as magnitude 4.5 or above.

The final saved model is a scikit-learn pipeline that includes preprocessing and the tuned classifier.

## Dataset

Source: [USGS Earthquake Hazards Program - all earthquakes, past month](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson)

The project uses a saved snapshot of the USGS GeoJSON feed. The snapshot was converted to CSV for easier analysis and reproducibility.

## Target

```text
is_strong_event
```

- `0`: magnitude below 4.5
- `1`: magnitude 4.5 or above

## Features

The model uses event location, depth, time, and compact metadata, including:

- latitude
- longitude
- depth
- absolute latitude
- log depth
- hour
- day of month
- felt reports
- tsunami flag
- USGS significance score
- magnitude type
- event status
- reporting network
- broad geographic quadrant

Raw magnitude and USGS significance score were not used as model inputs because they would leak the target definition.

## Models Compared

- Logistic Regression
- Random Forest
- Extra Trees
- Hist Gradient Boosting

The best baseline model was tuned with randomized search and cross-validation.

## Intended Use

This model is intended for educational data science and portfolio demonstration. It can help show how reported earthquake properties relate to stronger event classification.

## Limitations

- This is not an earthquake forecasting system.
- The model classifies already reported events from recorded properties.
- The dataset is a moving USGS monthly snapshot, so results can change when refreshed.
- The model does not include geological plate boundary data, historical fault-line information, or real-time sensor streams.

## Artifacts

- `models/best_earthquake_strength_model.joblib`
- `models/best_earthquake_strength_model_metadata.json`
- `outputs/baseline_model_results.csv`
- `outputs/tuned_model_results.csv`
- `outputs/feature_importance.csv`
