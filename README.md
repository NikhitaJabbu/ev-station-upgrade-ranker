# EV Charging Station Upgrade Predictor

> Which non-DC-Fast EV charging stations across the US are the best candidates for a DCFC upgrade?

A two-stage machine learning pipeline that scores and ranks 71,509 non-DCFC stations from the DOE Alternative Fuels Data Center dataset using a Random Forest classifier (CV Macro F1 = 0.976) and a four-signal composite priority index.

*CS490/590 Intensive Python for Data Science — SIUE, Spring 2026*

---

## Overview

EV infrastructure in the US is growing fast, but DC Fast Charging (DCFC) coverage is uneven. Most stations are Level-2 only. This project answers: **given what we know about a station's physical and geographic characteristics, how likely is it to become a DCFC station, and how urgently does it need one?**

The pipeline has three stages:

| Stage | What it does |
|-------|-------------|
| **1 — Classification** | Trains a Random Forest to predict DCFC presence; picks the best model by 5-fold CV |
| **2 — Priority Scoring** | Scores each non-DCFC station on upgrade probability, geographic scarcity, station age, and port count |
| **3 — Explainability** | Uses SHAP TreeExplainer to explain what drives each station's score |

---

## Results

### Model Comparison (5-fold Stratified CV)

| Model | CV Macro F1 | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.881 ± 0.005 | 0.953 ± 0.003 |
| XGBoost | 0.959 ± 0.002 | 0.996 ± 0.000 |
| **Random Forest** | **0.976 ± 0.002** | **0.996 ± 0.001** |

### Priority Scoring

- **14,302 stations** flagged as high-priority upgrade candidates (top 20% of 71,509 non-DCFC stations)
- **Geographic scarcity** (`nearby_dcfast_10km`) and **network type** were the strongest SHAP features
- **Rank stability**: mean Spearman ρ = 0.87 across 50 random weight sets — the top-ranked stations stay near the top even when weights are fully randomized

### Composite Score Weights

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Upgrade probability (`s_prob`) | 0.40 | Model confidence this station fits the DCFC profile |
| Geographic scarcity (`s_scarcity`) | 0.30 | Stations in underserved areas have more unmet need |
| Station age (`s_age`) | 0.15 | Older stations are more overdue for an upgrade |
| Port count (`s_ports`) | 0.15 | More ports = more demand = higher upgrade impact |

---

## Visualizations

All plots are saved to `outputs/` when the notebook runs.

| Plot | Description |
|------|-------------|
| `model_comparison.png` | CV F1-macro and ROC-AUC bar chart across three models |
| `confusion_matrix.png` | Test-set confusion matrix for the best model |
| `feature_importance.png` | XGBoost built-in gain-based feature importance |
| `shap_beeswarm.png` | Global SHAP beeswarm — which features push predictions and in which direction |
| `shap_waterfall_rank1/2/3.png` | Per-station SHAP waterfall for the top 3 ranked upgrade candidates |
| `priority_map.png` | US map of all non-DCFC stations colored by priority band |
| `priority_score_dist.png` | Priority score distribution by band (High / Medium / Low) |
| `correlation_heatmap.png` | Pearson correlation matrix of key numerical features |
| `station_age_dist.png` | Station age distribution, split by DCFC status |
| `class_distribution.png` | Class balance: 17.9% DCFC vs. 82.1% non-DCFC |

---

## Project Structure

```
capstone_v2.ipynb         Main notebook — EDA, modeling, scoring, SHAP
requirements.txt          Python dependencies
outputs/
  *.png                   All generated figures (16 plots)
  ranked_stations.csv     Full ranked list of 71,509 stations (generated; not tracked in git)
```

> **Note:** `ranked_stations.csv` (~14 MB) is excluded from git. It is generated when you run the notebook.

---

## Dataset

**DOE Alternative Fuels Data Center — Alternative Fuel Stations**
- Source: [https://afdc.energy.gov/stations](https://afdc.energy.gov/stations)
- 87,150 rows × 75 columns (one row per US alternative fueling station, April 2026 snapshot)

### How to download

1. Go to [https://afdc.energy.gov/data_download](https://afdc.energy.gov/data_download)
2. Under **Dataset**, select *Alternative Fuel Stations*
3. Fill in your name and email (required by the government portal)
4. Download the CSV and save it in the same directory as the notebook
5. Rename it to: `alt_fuel_stations (Apr 18 2026).csv`

   *(Or update the `CSV_FILE` path in Phase 1 of the notebook to match your filename)*

---

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook capstone_v2.ipynb
```

The notebook is self-contained and runs top to bottom. All outputs are written to the `outputs/` folder automatically.

**Python 3.10+ recommended.**

---

## Methodology

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `total_ports` | Level-1 + Level-2 port count |
| `station_age_years` | Days since `Open Date` / 365.25; missing filled with median |
| `nearby_dcfast_10km` | Count of existing DCFC stations within 10 km (haversine) |
| `EV Network_enc` | Label-encoded network operator |
| `Facility Type_enc` | Label-encoded facility type (74% missing → filled as `Unknown`) |
| `State_enc` | Label-encoded US state |
| `Access Code_enc` | Label-encoded public/private access code |

### Class Imbalance

17.9% of stations have DC Fast charging vs. 82.1% that do not (4.6:1 imbalance). SMOTE oversampling is applied **inside each CV fold** to prevent synthetic samples from leaking into validation folds.

### Spatial Scarcity

`nearby_dcfast_10km` is computed with a vectorized haversine formula against all 15,641 existing DCFC stations. This is the single strongest predictor — stations in underserved corridors with few nearby DCFC stations are consistently flagged as high-priority.

---

## Limitations

- Uses a single dataset snapshot; cannot predict future demand growth, NEVI funding decisions, or new highway corridors
- `Facility Type` is 74% missing — the feature is a weak placeholder for most stations
- Composite score weights are hand-tuned (no historical upgrade data to learn from); the sensitivity analysis shows rankings are stable but the weights remain a judgment call

---

## License

This project is for academic purposes. The dataset is from the US DOE AFDC and is publicly available.
