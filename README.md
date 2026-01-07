# Soft-Sensor Development for Ethanol Distillation Column

A comprehensive data-driven soft-sensing pipeline for predicting ethanol distillate purity in a binary distillation column, integrating **deep exploratory data analysis (EDA)**, **physically motivated feature engineering**, and **machine learning models**.

This repository is structured to clearly separate **exploratory analysis**, **EDA-driven insights**, and **final modeling**, reflecting an industrial soft-sensor development workflow.


## Repository Structure

MLPE-Project/
│── main.py                         # Integrated modeling pipeline (preprocessing + ML)
│── EDA_main.py                     # Standalone, detailed EDA script
│── EDA_Softsensing.pptx            # EDA summary & interpretation (presentation)
│── dataset_distill.csv             # Input dataset (user provided)
│── plots/                          # Auto-generated EDA & modeling figures
│── README.md                       # Documentation
└── MLPE Project Report.pdf         # Final project report


## Project Workflow

1. **Deep Exploratory Data Analysis (EDA)**
2. **EDA-driven feature engineering**
3. **Dynamic behavior analysis (autocorrelation & lag justification)**
4. **Regression, classification, and clustering models**
5. **Performance evaluation & visualization**


## Exploratory Data Analysis (EDA)

The EDA is implemented in **EDA_main.py** and summarized conceptually in **EDA_Softsensing.pptx**.

### Dataset Overview
- **Samples:** 4408  
- **Variables:** 21  
- **Target:** Ethanol concentration (distillate purity)
- **Inputs:**
  - Tray temperatures: T1–T14
  - Flow variables: L, V, D, B, F
  - Pressure (removed – constant)

### Data Cleaning
- Automatic numeric coercion of flow variables
- Removal of constant variables (e.g., pressure)
- NaN handling (EDA-only row removal for clean statistical analysis)

### Univariate Analysis
- Mean, median, variance, skewness, kurtosis
- Identification of:
  - Right-skewed flow variables
  - High-variance middle trays (T6–T10)
- Insight: **Middle trays are most composition-sensitive**

### Distribution & Outlier Analysis
- Histograms and boxplots for key variables
- Outliers treated as **process events**, not bad data
- Conclusion: Do not blindly remove outliers in industrial data

### Correlation & Mutual Information Analysis
- Pearson, Spearman, and Mutual Information (MI)
- Observations:
  - Strong negative correlation between ethanol purity and tray temperatures
  - MI decreases axially from top to bottom trays
- Physical explanation:
  - Higher ethanol → lower boiling point → lower tray temperatures

### Redundancy Analysis
- Strong correlation between adjacent trays
- Near-perfect correlation between L and V
- Mass balance coupling between F, D, and B
- Conclusion:
  - Raw variables are highly redundant
  - Feature compression is necessary


## Feature Engineering (EDA-Guided)

Implemented in both **EDA_main.py** (analysis) and **main.py** (modeling).

### Tray Compression
Representative trays selected:
- T2, T5, T8, T11, T14

### Temperature Difference Features
- ΔT(T2 − T14)
- ΔT(T2 − T8)
- ΔT(T8 − T14)

These act as surrogates for **composition gradients** and separation driving force.

### Ratio-Based Process Features
- L / D → Reflux intensity proxy  
- V / F → Vapor loading  
- D / F → Distillate recovery  

> Note: L/D is not the true reflux ratio but is used as a physically meaningful proxy.

### Validation
- Engineered features show:
  - High Mutual Information
  - Weak Pearson/Spearman correlation  
- Interpretation:
  - Relationship is **nonlinear and regime-dependent**
  - Linear-only models are insufficient


## Dynamic Behavior Analysis

- Ethanol concentration shows **long memory**
- Manipulated variables respond faster than composition
- Autocorrelation analysis confirms:
  - Strong process dynamics
  - Physical justification for **lag features**

Conclusion:
> Lagged inputs are essential for accurate soft-sensor modeling.


## Modeling (main.py)

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- MLP Regressor

**Outputs:**
- MAE, RMSE, R²
- True vs Predicted plots
- Parity plots

### Classification
- Quantile-based binning: Low / Medium / High purity
- Logistic Regression
- Confusion matrix & classification report

### Clustering
- KMeans on tray temperatures
- Automatic K selection using silhouette score
- Cluster-wise mean temperature profiles


## Key Takeaways

- Distillation behavior is **nonlinear, dynamic, and redundant**
- Physically motivated feature engineering outperforms raw variables
- EDA is not optional — it **defines the modeling strategy**
- Dynamic, nonlinear soft sensors are justified for this system


## Reference

Cote-Ballesteros, J. E., Grisales Palacios, V. H., & Rodriguez-Castellanos, J. E. (2022).  
Un algoritmo de selección de variables de enfoque híbrido basado en información mutua para aplicaciones de sensores blandos industriales basados en datos.  
*Ciencia e Ingeniería Neogranadina*, 32(1), 59–70.  
https://doi.org/10.18359/rcin.5644
