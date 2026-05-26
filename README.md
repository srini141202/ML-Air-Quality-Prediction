# ML Air Quality Prediction — PM2.5 in Dublin

> A Comparative and Explainable Study of Machine Learning and Deep Learning Models for Time-Series Air Quality Prediction in Dublin Using Multi-Pollutant Data

---

## Project Overview

This project compares four models — **Random Forest**, **XGBoost**, **LSTM**, and **MLP** — for predicting daily PM2.5 concentrations in Dublin using two years of multi-pollutant monitoring data (2011–2012) from Dublin City Council.

The project goes beyond accuracy metrics by incorporating:
- **SHAP explainability** on both tree-based models
- **Ablation study** to quantify the value of co-pollutant features
- **Error analysis** distinguishing peak vs non-peak event performance
- **Inference latency** benchmarking for deployment considerations

---

## Key Results

| Model | RMSE (µg/m³) | MAE | R² | Latency (ms) |
|---|---|---|---|---|
| Random Forest | **3.708** | 2.597 | **0.640** | 43.38 |
| XGBoost | 3.886 | 2.823 | 0.604 | **0.404** |
| LSTM | 7.444 | 5.115 | -0.430 | 76.863 |
| MLP | 7.063 | 3.600 | -0.308 | 0.315 |

- **Random Forest** achieved the best accuracy (R² = 0.640)
- **XGBoost** is the best deployment option (0.4 ms latency)
- **Deep learning models** underperformed due to limited training data (~500 sequences)
- Removing co-pollutant features from XGBoost increased RMSE by **27.3%**
- **SHAP** confirmed PM10 and NO2 as the dominant predictors in both tree-based models

---

## Dataset

- **Source:** Dublin City Council / [data.gov.ie](https://data.smartdublin.ie/dataset/air-quality-monitoring-data-dublin-city)
- **Period:** 2011–2012 (daily resolution)
- **Pollutants:** PM2.5, PM10, NO2, NO, SO2, CO
- **Monitoring stations:** 6 sites across Dublin
- **Records after cleaning:** 732 rows
- **Engineered features:** 19

---

## Project Structure

```
ML-Air-Quality-Prediction/
│
├── ML_CA2.ipynb              # Main notebook — full pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
│
├── data/                     # Raw CSV files (Dublin City Council)
│   ├── PM25_2011.csv
│   ├── PM25_2012.csv
│   ├── PM10_2011.csv
│   ├── PM10_2012.csv
│   ├── Gases_2011.csv
│   └── Gases_2012.csv
│
└── figures/                  # Generated plots (reproducible from notebook)
    ├── fig_timeseries.png
    ├── fig_correlation.png
    ├── fig_metrics.png
    ├── fig_shap_xgb_bar.png
    ├── fig_shap_xgb_beeswarm.png
    ├── fig_shap_rf.png
    ├── fig_predictions.png
    ├── fig_ablation.png
    ├── fig_residuals.png
    └── fig_learning_curve.png
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/srini141202/ML-Air-Quality-Prediction.git
cd ML-Air-Quality-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook ML_CA2.ipynb
```

Run all cells from top to bottom. The notebook handles all data loading, cleaning, feature engineering, model training, evaluation, SHAP analysis, ablation study, and figure generation.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
tensorflow
```

---

## Methodology Summary

| Step | Details |
|---|---|
| Data source | Dublin City Council open data (8 CSV files) |
| Cleaning | Clip negatives, winsorise NO outlier, forward-fill ≤3 days |
| Features | 19 engineered (lags, rolling stats, co-pollutant lags, temporal) |
| Split | 70% train / 15% val / 15% test — chronological, no shuffle |
| Scaling | MinMaxScaler (trees) · StandardScaler + PCA (MLP) |
| Evaluation | RMSE, MAE, R², training time, inference latency |
| Explainability | SHAP TreeExplainer on RF and XGBoost (109 test samples) |
| Ablation | XGBoost: full 19 features vs PM2.5-only features |

---

## Research Questions

1. By how much do Random Forest, XGBoost, LSTM, and MLP differ in predicting daily PM2.5 concentrations, as measured by RMSE, MAE, and R²?
2. How much does removing co-pollutant features from XGBoost increase RMSE relative to the full 19-feature set?
3. Which features carry the highest SHAP attribution values, and do rankings agree across both tree-based models?

---

## References

Key references used in this project:

- Garbagna et al. (2025) — AI-driven approaches for air pollution modelling, *Environmental Pollution*
- Eren et al. (2023) — LSTM for PM2.5 prediction in Istanbul, *Urban Climate*
- Samad et al. (2023) — Multi-pollutant ML for virtual monitoring stations, *Atmospheric Environment*
- Lundberg & Lee (2017) — SHAP: A unified approach to interpreting model predictions, *NeurIPS*
- Joharestani et al. (2019) — PM2.5 prediction using RF, XGBoost and DL, *Atmosphere*

---
