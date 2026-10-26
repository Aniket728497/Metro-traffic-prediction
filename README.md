<div align="center">

# Metro Interstate Traffic Volume Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-0052CC?style=for-the-badge&logo=archive&logoColor=white)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> An end-to-end machine learning pipeline to predict hourly I-94 traffic volume from meteorological and temporal features, featuring rigorous multi-stage preprocessing, anomaly cleaning, and a fully optimized Random Forest regressor.

</div>

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Project Workflow & Preprocessing](#-project-workflow--preprocessing)
- [Model Evaluation & Progression](#-model-evaluation--progression)
- [Final Model Performance](#-final-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## About the Project

Accurate traffic volume forecasting is vital for urban planning, congestion mitigation, and intelligent transportation systems. This project implements a robust, end-to-end machine learning pipeline using the **Metro Interstate Traffic Volume** dataset. 

To ensure high generalization capability and prevent data leakage, the project incorporates a strict three-way data partition (**70% Train / 15% Validation / 15% Test**), leakage-free feature scaling using `StandardScaler`, and automated hyperparameter optimization.

**Key Highlights:**
- **Data Hygiene:** Cleaned 48,204 raw records by removing corrupted `0 K` temperature readings and verified highway closure/sensor outages (`traffic_volume <= 110`).
- **Feature Engineering:** Extracted cyclical temporal features (`hour`, `day_of_week`, `month`), propagated midnight holiday markers across full 24-hour cycles, and consolidated granular weather descriptions into distinct binary severity flags.
- **Modeling Excellence:** Overcame linear assumptions by transitioning to a heavily regularized Random Forest model, achieving an exceptional **Test $R^2$ of 0.9654**.

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)[cite: 4] |
| **Original Size** | 48,204 hourly records across 9 columns |
| **Cleaned Size** | 40,517 valid rows across 14 refined numerical features |

---

## Project Workflow & Preprocessing

The pipeline cleans and transforms the raw data through four meticulous stages:

1. **Holiday Propagation & Encoding:** Propagated midnight holiday labels across the full 24 hours of each calendar day, categorizing them into `Major_Holiday`, `Minor_Holiday`, `State_Fair`, and `Normal_Day` (one-hot encoded).
2. **Weather Categorization:** Mapped dozens of text descriptions into clean meteorological severity indicators (`fog`, `light_rain`, `heavy_rain`, `light_snow`, `heavy_snow`, `storm`) while dropping redundant text and cloud cover columns.
3. **Timestamp Aggregation:** Resolved duplicate hourly timestamps caused by simultaneous weather logging by computing means for continuous features and maximums for binary flags.
4. **Sensor Anomaly Filtering:** Removed 10 corrupted `0 K` temperature rows and 48 near-zero traffic records (`traffic_volume <= 110`) representing sensor outages or temporary road closures.

---

## Model Evaluation & Progression

Performance across key development milestones (evaluated on training and validation splits):

| Model Stage | Split | MAE (cars) | RMSE (cars) | $R^2$ |
|-------------|-------|:----------:|:-----------:|:-----:|
| **1. Default Random Forest** | Training | 92.76 | 149.13 | 0.9943 |
| **2. Default Random Forest** | Validation | 246.04 | 384.42 | 0.9621 |
| **3. Manually Tuned RF** | Training | 178.45 | 290.53 | 0.9785 |
| **4. Manually Tuned RF** | Validation | 235.13 | 370.44 | 0.9648 |
| **5. Automatically Tuned RF** | Validation | 227.33 | 355.06 | 0.9676 |


---

## Final Model Performance

The finalized Random Forest model was evaluated once on the strictly held-out test set:

| Metric | Test Set Score |
|--------|:--------------:|
| **Mean Absolute Error (MAE)** | **231.18 cars** |
| **Root Mean Squared Error (RMSE)** | **373.50 cars** |
| **Coefficient of Determination ($R^2$)** | **0.9654** |

---

## Sample Predictions

```python
# Input order: temp(K), hour, day_of_week, month, followed by 9 binary flags 
# (Major_Hol, Minor_Hol, State_Fair, fog, h_rain, h_snow, l_rain, l_snow, storm)

print(f"-> Predicted traffic: {predict_traffic(298.0, 17, 1, 7, 0,0,0,0,0,0,0,0,0):.2f} cars/hour")
# → Predicted traffic volume is ~6,071 cars/hour (Peak evening rush hour on a warm summer Tuesday)

print(f"-> Predicted traffic: {predict_traffic(265.0, 3, 6, 1, 0,0,0,0,0,0,0,1,1):.2f} cars/hour")
# → Predicted traffic volume is ~411 cars/hour (Snowy overnight traffic on a winter Sunday)```
```
---

## Project Structure

```text
Metro-Interstate-Traffic-Volume-Prediction/
│
├── Metro_Traffic_Prediction.ipynb       # Main end-to-end Jupyter Notebook
│                                          (EDA, preprocessing, tuning, visualizations)
├── Metro_Interstate_Traffic_Volume.csv  # Raw dataset
├── requirements.txt                     # Python dependencies
├── .gitignore                            # Git ignore rules for cache and system files
└── README.md                             # Project documentation
```

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Aniket728497/Metro-traffic-prediction.git

cd metro-traffic-prediction
```

### 2. Set Up a Virtual Environment and Install Dependencies

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Launch the Jupyter Notebook

Ensure that `Metro_Interstate_Traffic_Volume.csv` is located in the root directory alongside the Jupyter Notebook.

Launch the notebook using:

```bash
jupyter notebook Metro_Traffic_Prediction.ipynb
```

## Tech Stack

| Category                        | Technologies            |
| ------------------------------- | ----------------------- |
| **Programming Language**        | Python 3.8+             |
| **Data Manipulation**           | Pandas, NumPy           |
| **Data Visualization**          | Matplotlib, Seaborn     |
| **Machine Learning**            | scikit-learn            |
| **Machine Learning Algorithms** | `RandomForestRegressor` |
| **Feature Scaling**             | `StandardScaler`        |
| **Dataset Splitting**           | `train_test_split`      |
| **Hyperparameter Tuning**       | `RandomizedSearchCV`    |
| **Development Environment**     | Jupyter Notebook        |
