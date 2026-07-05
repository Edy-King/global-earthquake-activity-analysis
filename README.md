# Global Earthquake Activity Analysis and Strong-Event Prediction

This project is my end-to-end data science analysis of recent global earthquake activity using data from the United States Geological Survey (USGS).

I explored where earthquakes happened, how magnitude and depth behaved, which regions had stronger activity, and whether a machine learning model could classify stronger earthquake events.

Dataset source: [USGS Earthquake Hazards Program - all earthquakes, past month](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson)

## Project Files

- `earthquake_activity_analysis.ipynb` - executed notebook with analysis, visualizations, model comparison, tuning, and saved-model workflow.
- `data/usgs_earthquakes_all_month.csv` - cleaned CSV snapshot created from the USGS GeoJSON feed.
- `data/usgs_earthquakes_all_month.geojson` - original downloaded GeoJSON snapshot.
- `data/dataset_metadata.json` - download metadata and source URL.
- `outputs/baseline_model_results.csv` - baseline model comparison.
- `outputs/tuned_model_results.csv` - tuned model results.
- `outputs/feature_importance.csv` - final model feature importance, when available.
- `models/best_earthquake_strength_model.joblib` - saved final model pipeline.
- `models/best_earthquake_strength_model_metadata.json` - final model metadata and metrics.
- `requirements.txt` - required Python packages.

## Main Questions

- Where did recent earthquakes happen globally?
- How are earthquake magnitude and depth distributed?
- Which broad regions had higher shares of stronger events?
- Which machine learning model performed best for classifying stronger earthquakes?
- Which features mattered most to the final model?

I deliberately did not use raw `magnitude` or USGS `sig` as model features because they would leak the target definition.

## Target Variable

I created a binary target called:

```text
is_strong_event
```

where:

- `0` means magnitude below 4.5
- `1` means magnitude 4.5 or above

This threshold gives the project a practical classification target while keeping the analysis simple and explainable.

## Models Compared

I trained and compared:

- Logistic Regression
- Random Forest
- Extra Trees
- Hist Gradient Boosting

The best baseline model was tuned using randomized hyperparameter search.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

ROC-AUC was used as the main ranking metric because stronger events are less common than lower-magnitude events.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Open and run:

```text
earthquake_activity_analysis.ipynb
```

The notebook uses the CSV snapshot in the `data` folder. To refresh the dataset, download the current USGS feed:

```text
https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson
```

## Notes

- The dataset is a time-based snapshot, so results will change if the data is refreshed.
- This model is for data science learning and portfolio demonstration, not earthquake forecasting.
- The model classifies reported events from their recorded properties; it does not predict future earthquakes.
