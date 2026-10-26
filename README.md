# Metro Interstate Traffic Volume Prediction

A machine-learning project that estimates traffic volume using traffic, weather, and calendar features. The full workflow is documented in [`Metro_Traffic_Prediction.ipynb`](Metro_Traffic_Prediction.ipynb).

## Project contents

- `Metro_Traffic_Prediction.ipynb` — end-to-end workflow, from data inspection and preprocessing to Random Forest model evaluation.
- `Metro_Interstate_Traffic_Volume.csv` — the original dataset (add this file next to the notebook; it is not included in this starter bundle).
- `requirements.txt` — Python packages needed to run the notebook.

The notebook consolidates the preprocessing stages in memory. You do not need to maintain separate intermediate `modified_v1.csv`–`modified_v4.csv` files.

## How to run

1. Put `Metro_Interstate_Traffic_Volume.csv` in this repository's root directory, alongside the notebook.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Start Jupyter: `jupyter notebook`.
4. Open `Metro_Traffic_Prediction.ipynb` and choose **Run All**.
5. Save the notebook after it has run so GitHub can display its cell outputs and plots.

## Methodology notes

The notebook preserves the main logic and Random Forest hyperparameters from the original scripts. It also highlights modelling choices that warrant review, particularly filtering out traffic volumes at or below 110 and using a random rather than chronological split. Review those choices before presenting the metrics as evidence of future-traffic forecasting performance.

## Dependencies

See [`requirements.txt`](requirements.txt).
