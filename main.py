#!/usr/bin/env python3


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix, classification_report,
                             silhouette_score)

# ---------------- CONFIG ----------------
DATA_PATHS = ["./dataset_distill.csv"]
RANDOM_STATE = 42

REGRESSORS = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001, max_iter=5000),
    'MLP': MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=RANDOM_STATE)
}
# Only logistic regression per request
CLASSIFIERS = {
    'LogReg': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
}

# Range of K to try for silhouette selection
K_MIN = 2
K_MAX = 10  # will be clipped versus n_samples

# choose which process vars to always include:
process_vars = ['Pressure', 'L', 'V', 'D', 'B', 'F', 'reflux_ratio']

# how many top correlated features (besides process_vars) to include
top_n = 6

# figure resolution
DPI = 250
CELL_SIZE = 2.0   # inches per variable cell (increase for larger text/readability)

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def savefig_and_close(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved plot -> {path}")

# ---------------- END CONFIG ----------------

def find_dataset(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def try_read_csv(path):
    for sep in [",",";","\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                print(f"Loaded file '{path}' with sep='{sep}' -> shape {df.shape}")
                return df
        except Exception:
            continue
    raise ValueError(f"Unable to parse CSV at {path} with common separators.")

# ---------------- Load ----------------
data_path = find_dataset(DATA_PATHS)
if data_path is None:
    raise FileNotFoundError("Dataset not found. Place 'dataset_distill.csv' in the working folder.")
df = try_read_csv(data_path)
df.columns = [str(c).strip() for c in df.columns]
print("Using dataset:", data_path)

# ---------------- Basic coercion & time index ----------------
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

if 'time' not in df.columns and 'time_h' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'step'})
    df['time_h'] = df['step'] * 0.1
else:
    if 'time_h' not in df.columns:
        df['time_h'] = df.get('time')

# Detect ethanol target
target_candidates = [c for c in df.columns if 'ethanol' in str(c).lower()]
if target_candidates:
    target_col = target_candidates[0]
else:
    target_col = df.columns[-1]
print("Target column detected:", target_col)

# Derived reflux_ratio if available
if {'L','V'}.issubset(df.columns):
    df['reflux_ratio'] = df['L'] / (df['V'] + 1e-9)

# Interpolate numeric columns, fill remaining with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both', axis=0)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Drop constant cols
const_cols = [c for c in num_cols if df[c].nunique() <= 1]
if const_cols:
    print("Dropping constant columns:", const_cols)
    df = df.drop(columns=const_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Detect temperature columns (T1..Tn)
temp_cols = [c for c in df.columns if str(c).strip().upper().startswith("T")]


# ---------------- Feature engineering ----------------
# temperature diffs
for i in range(max(0, len(temp_cols)-1)):
    a = temp_cols[i]; b = temp_cols[i+1]
    df[f"dT_{i+1}_{i+2}"] = df[b] - df[a]
if len(temp_cols) > 0:
    df['T_range'] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)
    df['T_mean'] = df[temp_cols].mean(axis=1)

# lag features for ethanol (1..5)
for lag in range(1,6):
    df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

# drop rows with NaNs from lagging
df = df.dropna().reset_index(drop=True)

# prepare feature matrix
exclude = ['step','time_h']
feature_columns = [c for c in df.columns if c not in exclude + [target_col]]
feature_columns = [c for c in feature_columns if np.issubdtype(df[c].dtype, np.number)]
X_all = df[feature_columns].copy()
y_reg = df[target_col].copy()

print(f"Prepared features: {len(feature_columns)} features, {len(df)} rows after lagging.")

# ---------------- EDA ----------------

# compute absolute correlations w.r.t target and pick top features
corr_full = df[num_cols].corr(method='pearson')
corr_with_target = corr_full[target_col].abs().sort_values(ascending=False).drop(target_col, errors='ignore')
top_feats = [c for c in corr_with_target.index if c not in process_vars][:top_n]

# selected features = process_vars + top_feats + target (ensure existence)
selected = []
for v in process_vars + top_feats + [target_col]:
    if v in df.columns and v not in selected:
        selected.append(v)

if len(selected) < 3:
    # fallback: pick first few numeric columns
    fallback = [c for c in num_cols if c not in selected]
    selected += fallback[:(3-len(selected))]

print("Selected features (count={}):".format(len(selected)))
print(selected)

plot_df = df[selected].copy()

# Prepare figure grid (n x n axes)
n = len(selected)
fig_w = CELL_SIZE * n
fig_h = CELL_SIZE * n
fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
# Create an n x n grid of axes
axes = [[fig.add_subplot(n, n, i * n + j + 1) for j in range(n)] for i in range(n)]

# Prepare colormap for correlation tiles
cmap = plt.get_cmap("coolwarm")
norm = Normalize(vmin=-1, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# Determine axis limits for each variable to share across row/col for consistent plotting
var_limits = {}
for col in selected:
    series = plot_df[col].dropna()
    if len(series) == 0:
        var_limits[col] = (0, 1)
    else:
        lo, hi = series.min(), series.max()
        # if lo == hi, expand a little so histogram/points show
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        rng = hi - lo
        var_limits[col] = (lo - 0.03 * rng, hi + 0.03 * rng)

# Fill the grid:
for i in range(n):
    for j in range(n):
        ax = axes[i][j]
        x_var = selected[j]   # column is x
        y_var = selected[i]   # row is y

        # Upper triangle (j > i): correlation tiles with numeric r
        if j > i:
            x = plot_df[x_var]
            y = plot_df[y_var]
            # compute Pearson r
            try:
                r = np.corrcoef(x.dropna(), y.dropna())[0,1]
            except Exception:
                r = np.nan
            color = cmap(norm(r if not np.isnan(r) else 0.0))
            ax.add_patch(patches.Rectangle((0,0),1,1, transform=ax.transAxes, color=color, alpha=0.6))
            ax.text(0.5, 0.5, f"{r:.2f}" if not np.isnan(r) else "nan", ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        # Diagonal (i == j): histogram
        elif i == j:
            series = plot_df[x_var].dropna()
            ax.hist(series, bins=24, alpha=0.9)
            ax.set_xlim(var_limits[x_var])
            ax.set_yticks([])
            # draw a small title (variable name) above histogram
            ax.set_title(x_var, fontsize=10, pad=6)

        # Lower triangle (j < i): scatter
        else:
            x = plot_df[x_var]
            y = plot_df[y_var]
            ax.scatter(x, y, s=6, alpha=0.6)
            ax.set_xlim(var_limits[x_var])
            ax.set_ylim(var_limits[y_var])

        # Only label left-most y axes and bottom-most x axes to avoid clutter
        if j != 0:
            ax.set_yticklabels([])
        if i != n-1:
            ax.set_xticklabels([])

        # put thin spines for neatness
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

# Set tick labels for outer axes
# Bottom row: x labels
for j in range(n):
    ax = axes[n-1][j]
    ax.set_xlabel(selected[j], fontsize=9, labelpad=6)
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)
# Left column: y labels
for i in range(n):
    ax = axes[i][0]
    ax.set_ylabel(selected[i], fontsize=9, labelpad=6)
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', labelsize=7)

# Add a colorbar on the right (for correlation tiles)
cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])  # left, bottom, width, height
cb = plt.colorbar(sm, cax=cax)
cb.set_label('Pearson r', rotation=270, labelpad=12)

# Final adjustments and save
plt.suptitle("Pearson Correlation Findings (Process vars + top correlated features)", fontsize=18, y=0.995)
plt.subplots_adjust(left=0.06, right=0.90, top=0.96, bottom=0.06, wspace=0.12, hspace=0.12)

savefig_and_close("EDA.png")



# ---------------- Regression ----------------
split_idx = int(0.8 * len(X_all))
X_train_reg = X_all.iloc[:split_idx]; X_test_reg = X_all.iloc[split_idx:]
y_train_reg = y_reg.iloc[:split_idx]; y_test_reg = y_reg.iloc[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_reg)
X_test_s = scaler.transform(X_test_reg)

reg_results = []
trained_regressors = {}
for name, mdl in REGRESSORS.items():
    print(f"Training regressor: {name}")
    mdl.fit(X_train_s, y_train_reg)
    preds = mdl.predict(X_test_s)
    mae = mean_absolute_error(y_test_reg, preds)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
    r2 = r2_score(y_test_reg, preds)
    reg_results.append({'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    trained_regressors[name] = mdl
    print(f" -> {name}: MAE={mae:.5f}, RMSE={rmse:.5f}, R2={r2:.5f}")

print("\nRegression results:")
print(pd.DataFrame(reg_results).sort_values('RMSE'))

# Visualize best regressor results
best_reg = sorted(reg_results, key=lambda r: r['RMSE'])[0]['model']
best_model_reg = trained_regressors[best_reg]
y_pred_best = best_model_reg.predict(X_test_s)

plt.figure(figsize=(14,3))
plt.plot(df['time_h'].iloc[split_idx:], y_test_reg.values, label='True', linewidth=1)
plt.plot(df['time_h'].iloc[split_idx:], y_pred_best, '--', label=f'Predicted ({best_reg})', linewidth=1)
plt.xlabel("Time (h)"); plt.ylabel(target_col); plt.legend()
plt.title(f"True vs Predicted ({best_reg})")
savefig_and_close("True vs Predicted ")

plt.figure(figsize=(5,5))
plt.scatter(y_test_reg.values, y_pred_best, s=8, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=1)
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"Parity plot ({best_reg})")
savefig_and_close("Parity plot ")

# ---------------- Classification ----------------
# keep quantile bins (Low/Med/High) as before
df['ethanol_class'] = pd.qcut(df[target_col], q=3, labels=[0,1,2])
y_clf = df['ethanol_class'].astype(int)

X_train_clf = X_all.iloc[:split_idx]; X_test_clf = X_all.iloc[split_idx:]
y_train_clf = y_clf.iloc[:split_idx]; y_test_clf = y_clf.iloc[split_idx:]

scaler_clf = StandardScaler()
X_train_clf_s = scaler_clf.fit_transform(X_train_clf)
X_test_clf_s = scaler_clf.transform(X_test_clf)

clf_results = []
trained_clfs = {}
for name, clf in CLASSIFIERS.items():
    print(f"Training classifier: {name}")
    clf.fit(X_train_clf_s, y_train_clf)
    y_pred_clf = clf.predict(X_test_clf_s)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    clf_results.append({'classifier': name, 'accuracy': acc })
    trained_clfs[name] = clf
    print(f" -> {name}: accuracy={acc:.4f}")



# Confusion matrix for the classifier
best_clf_name = sorted(clf_results, key=lambda r: r['accuracy'], reverse=True)[0]['classifier']
best_clf = trained_clfs[best_clf_name]
y_pred_best_clf = best_clf.predict(X_test_clf_s)
cm = confusion_matrix(y_test_clf, y_pred_best_clf)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix ({best_clf_name})")
savefig_and_close("Confusion Matrix ({best_clf_name})")

print("\nClassification report (best classifier):")
print(classification_report(y_test_clf, y_pred_best_clf))

# ---------------- Temperature clustering  ----------------
if len(temp_cols) >= 2:
    print("\nRunning KMeans clustering on temperature columns")
    X_temp = df[temp_cols].values
    X_temp = np.nan_to_num(X_temp)
    scaler_temp = StandardScaler()
    X_temp_s = scaler_temp.fit_transform(X_temp)

    # define K range according to data size
    n_samples = X_temp_s.shape[0]
    k_max_allowed = min(K_MAX, n_samples - 1)
    k_min_allowed = max(K_MIN, 2)
    if k_max_allowed < k_min_allowed:
        print("Not enough samples to run silhouette-based K selection. Skipping clustering.")
    else:
        silhouette_scores = []
        k_values = list(range(k_min_allowed, k_max_allowed + 1))
        for k in k_values:
            kmeans_try = KMeans(n_clusters=k, random_state=RANDOM_STATE)
            labels_try = kmeans_try.fit_predict(X_temp_s)
            # silhouette requires at least 2 clusters and less than n_samples clusters
            try:
                sil = silhouette_score(X_temp_s, labels_try)
            except Exception:
                sil = -1
            silhouette_scores.append(sil)
            print(f"Attempted K={k}, silhouette={sil:.4f}")

        # plot silhouette vs K
        plt.figure(figsize=(8,4))
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.xlabel("k (number of clusters)"); plt.ylabel("Silhouette score")
        plt.title("Silhouette score vs k (temperature clustering)")
        plt.grid(True)
        savefig_and_close("Silhouette score vs k (temperature clustering)")

        # pick best k (max silhouette). If tie, choose smaller k.
        best_idx = int(np.argmax(silhouette_scores))
        best_k = k_values[best_idx]
        best_sil = silhouette_scores[best_idx]
        print(f"Selected best_k={best_k} with silhouette={best_sil:.4f}")

        # Fit final KMeans with best_k
        kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE)
        temp_labels = kmeans.fit_predict(X_temp_s)
        df['temp_cluster'] = temp_labels

       

        # Visual 2: mean temperature profile per cluster
        plt.figure(figsize=(8,5))
        trays = np.arange(1, len(temp_cols)+1)
        for cl in sorted(np.unique(temp_labels)):
            mean_profile = X_temp[temp_labels == cl].mean(axis=0)
            plt.plot(trays, mean_profile, marker='o', label=f'cluster {cl}')
        plt.xlabel("Tray index"); plt.ylabel("Temperature")
        plt.title("Mean temperature profile per cluster")
        plt.legend()
        savefig_and_close("Mean temperature profile per cluster")


else:
    print("Not enough temperature columns to run temperature-only clustering (need >=2).")


