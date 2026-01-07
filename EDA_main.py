import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from pandas.plotting import autocorrelation_plot

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

from scipy.stats import skew, kurtosis

# Load dataset
df = pd.read_csv("dataset_distill.csv", sep=";")

print("Initial shape:", df.shape)

# Strip column names (safety)
df.columns = df.columns.str.strip()

# Convert flow columns to numeric
for col in ["L", "V", "D", "B", "F"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("NaNs after type coercion:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# Drop constant columns
variance = df.var(numeric_only=True)
constant_cols = variance[variance == 0].index.tolist()

print("Dropping constant columns:", constant_cols)
df = df.drop(columns=constant_cols)

print("Final shape after cleaning:", df.shape)
#print(df.dtypes)


# Drop rows with NaNs (EDA-only)
initial_rows = df.shape[0]
df = df.dropna()
final_rows = df.shape[0]

print(f"Dropped {initial_rows - final_rows} rows due to NaNs")

#Step 2

numeric_cols = df.columns  # all are numeric now

univariate_stats = pd.DataFrame(index=numeric_cols)

univariate_stats["mean"] = df.mean()
univariate_stats["median"] = df.median()
univariate_stats["std"] = df.std()
univariate_stats["var"] = df.var()
univariate_stats["min"] = df.min()
univariate_stats["max"] = df.max()
univariate_stats["skewness"] = df.apply(lambda x: skew(x), axis=0)
univariate_stats["kurtosis"] = df.apply(lambda x: kurtosis(x), axis=0)

# Sort by variance (very useful for tray analysis)
univariate_stats = univariate_stats.sort_values("var", ascending=False)

print(univariate_stats)

#Step 3 
sns.set_style("whitegrid")

vars_to_plot = [
    "Ethanol concentration",
    "L",
    "V",
    "T1",
    "T7",
    "T14"
]
#histograms
for col in vars_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=40, kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    fname = col.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join(PLOT_DIR, f"hist_{fname}.png"),
        dpi=300,
        bbox_inches="tight"
    )   
#boxplots
for col in vars_to_plot:
    plt.figure(figsize=(6,2.5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()

    fname = col.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join(PLOT_DIR, f"box_{fname}.png"),
        dpi=300,
        bbox_inches="tight"
    )

#Step 4 
target = "Ethanol concentration"

X = df.drop(columns=[target])
y = df[target]

results = []

for col in X.columns:
    pearson = np.corrcoef(X[col], y)[0, 1]
    spearman = spearmanr(X[col], y).correlation
    mi = mutual_info_regression(
        X[[col]], y, random_state=42
    )[0]

    results.append([col, pearson, spearman, mi])

corr_mi_df = pd.DataFrame(
    results,
    columns=["Variable", "Pearson", "Spearman", "Mutual_Information"]
).sort_values("Mutual_Information", ascending=False)

print(corr_mi_df)

#Step 5 
X_inputs = df

plt.figure(figsize=(14, 10))
sns.heatmap(
    mi,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)
plt.title("Input–Input Correlation Heatmap (Redundancy Analysis)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "input_input_correlation_heatmap.png"), dpi=300)
 
#Step 6
temp_features = ["T2", "T5", "T8", "T11", "T14"]
df["dT_top_bottom"] = df["T2"] - df["T14"]
df["dT_top_middle"] = df["T2"] - df["T8"]
df["dT_middle_bottom"] = df["T8"] - df["T14"]
df["reflux_ratio_proxy"] = df["L"] / df["D"]
df["vapor_feed_ratio"] = df["V"] / df["F"]
df["distillate_recovery"] = df["D"] / df["F"]
engineered_features = (
    temp_features +
    [
        "dT_top_bottom",
        "dT_top_middle",
        "dT_middle_bottom",
        "reflux_ratio_proxy",
        "vapor_feed_ratio",
        "distillate_recovery"
    ]
)
X_eng = df[engineered_features]
y = df["Ethanol concentration"]

print("Engineered feature set:")
print(X_eng.columns.tolist())
print("Shape:", X_eng.shape)

#Step 7
X_eng = df[engineered_features]
results_eng = []

for col in X_eng.columns:
    pearson = np.corrcoef(X_eng[col], y)[0, 1]
    spearman = spearmanr(X_eng[col], y).correlation
    mi = mutual_info_regression(X_eng[[col]], y, random_state=42)[0]

    results_eng.append([col, pearson, spearman, mi])

corr_mi_eng_df = pd.DataFrame(
    results_eng,
    columns=["Feature", "Pearson", "Spearman", "Mutual_Information"]
).sort_values("Mutual_Information", ascending=False)

print("\nENGINEERED FEATURE VALIDATION (MI-based):\n")
print(corr_mi_eng_df)

#Step 8
plt.figure(figsize=(12,4))
plt.plot(df["Ethanol concentration"].values)
plt.title("Ethanol Concentration vs Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("Ethanol Concentration")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "time_evaluation.png"), dpi=300)

plt.figure(figsize=(6,4))
autocorrelation_plot(df["Ethanol concentration"])
plt.title("Autocorrelation of Ethanol Concentration")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "autocorrection.png"), dpi=300)

features_to_check = [
    "T2",
    "T8",
    "dT_top_bottom",
    "vapor_feed_ratio",
    "reflux_ratio_proxy"
]

for col in features_to_check:
    plt.figure(figsize=(6,4))
    autocorrelation_plot(df[col])
    plt.title(f"Autocorrelation of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "imp_autocorrection.png"), dpi=300)
