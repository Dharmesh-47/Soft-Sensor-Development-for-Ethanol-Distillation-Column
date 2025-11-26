# Soft-Sensor Development for Ethanol Distillation Column

A complete pipeline for data cleaning, feature engineering, EDA visualization, regression, classification, and clustering on a distillation column dataset.

This repository contains an integrated script (main.py) that loads the dataset, performs automated preprocessing, generates analytical plots, trains models, evaluates them, and saves all output figures into a dedicated folder.

<pre>
MLPE-Project/
│── main.py                         # Combined modeling + EDA pipeline
│── dataset_distill.csv             # Input dataset (user provided)
│── plots/                          # Auto-generated output figures (created at runtime)
│── README.md                       # Documentation
└── MLPE Project Report.pdf         # Project report
</pre>

## Features
### Automatic Data Loading

- Reads dataset_distill.csv with multiple fallback separators.
- Auto-detects ethanol target column.
- Handles missing values by interpolation + median filling.
- Drops constant columns.

### Feature Engineering

- Temperature differences (dT_i), mean, range.
- Lag features for ethanol (lag1…lag5).
- Derived reflux_ratio if L and V exist.

### Exploratory Data Analysis

- Generates a combined PairGrid-like correlation plot including:
- histograms (diagonal)
- scatterplots (lower triangle)
- correlation tiles (upper triangle)
- Auto-selects most relevant features.
- Saves output as plots/EDA.png.

### Regression Models

- Trains and evaluates:
- - Linear Regression
- - Ridge
- - Lasso
- - MLP Regressor
- Outputs:
- - MAE, RMSE, R² tables
- - True vs Predicted plot
- - Parity plot

### Classification

- Quantile binning of target into Low/Medium/High classes.
- Logistic Regression classifier.
- Confusion matrix + classification report.

## Reference
Cote-Ballesteros, J. E., Grisales Palacios, V. H., & Rodriguez-Castellanos, J. E. (2022).
Un algoritmo de selección de variables de enfoque híbrido basado en información mutua para aplicaciones de sensores blandos industriales basados en datos.
Ciencia e Ingeniería Neogranadina, 32(1), 59–70.
https://doi.org/10.18359/rcin.5644

### Clustering

- KMeans on temperature columns.
- Silhouette-based automatic K selection.
- Saves silhouette vs. K curve.
- Saves mean temperature profile per cluster.
