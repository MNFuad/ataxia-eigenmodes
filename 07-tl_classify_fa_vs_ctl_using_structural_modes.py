#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tl_classify_fa_vs_ctl_using_structural_modes.py

Purpose
-------
Transfer-learning style classifier using ONLY structural eigenmode
weights to distinguish FA patients from healthy controls.

This script performs:
  • Load precomputed structural eigenmode weights per subject
  • Train/test using Leave-One-Out Cross-Validation (LOOCV)
  • t-test feature selection inside each fold (train set only)
  • Neural-network classifier with early stopping
  • Outputs performance metrics + ROC curve

Expected Input Files
--------------------
All inputs must reside under:

    repo_root/data/transfer_learning/structural/

1) structural_modes.mat   (MATLAB file)
   - Contains a struct or dict with eigenmode weights, e.g.:
       MBM.eig.beta_subject   [nModes x nSubjects] or [nSubjects x nModes]
   - This script assumes [nModes x nSubjects] and transposes to
     [nSubjects x nModes]. If your file is already [nSubjects x nModes],
     you can adapt the loading line accordingly.

2) group_labels.csv
   - CSV with at least the following column:
       group_numeric
     where:
       0 = healthy control
       1 = FA

   - Optionally, you may include:
       subject_id
     but it is not strictly required by this script.

Directory Layout Example
------------------------
repo_root/
  data/
    transfer_learning/
      structural/
        structural_modes.mat
        group_labels.csv

Outputs
-------
All outputs are saved under:

    repo_root/results/transfer_learning/
        tl_classify_fa_vs_ctl_using_structural_modes/

Files generated:
  • metrics.csv
      - accuracy, precision, recall, f1, auc
  • roc_curve.png
      - ROC curve over the full LOOCV

Notes
-----
- This script does NOT generate any data by itself.
- It is generic and can be used with any dataset that matches the
  file format described above.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_ind

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt


# -----------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------
SEED = 44
np.random.seed(SEED)
tf.random.set_seed(SEED)



def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=2000, random_state=44):
    """Return percentile bootstrap 95% CI for ROC AUC.

    Bootstrap resamples that contain only one class are skipped because
    roc_auc_score is undefined for a single-class sample.
    """
    y_true_np = np.asarray(y_true, dtype=int)
    y_scores_np = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true_np)) < 2:
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    bootstrapped_auc = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_np), len(y_true_np))
        if len(np.unique(y_true_np[indices])) < 2:
            continue
        auc_i = roc_auc_score(y_true_np[indices], y_scores_np[indices])
        bootstrapped_auc.append(auc_i)

    if len(bootstrapped_auc) == 0:
        return np.nan, np.nan

    ci_lower = np.percentile(bootstrapped_auc, 2.5)
    ci_upper = np.percentile(bootstrapped_auc, 97.5)
    return ci_lower, ci_upper


def main():
    # -------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------
    # Assuming: repo_root/transfer_learning/this_script.py
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "transfer_learning" / "structural"
    results_dir = (
        repo_root
        / "results"
        / "transfer_learning"
        / "tl_classify_fa_vs_ctl_using_structural_modes"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    structural_mat_file = data_dir / "structural_modes.mat"
    group_file = data_dir / "group_labels.csv"

    if not structural_mat_file.exists():
        raise FileNotFoundError(f"Missing input file: {structural_mat_file}")
    if not group_file.exists():
        raise FileNotFoundError(f"Missing group label file: {group_file}")

    # -------------------------------------------------------
    # 1. Load structural eigenmode data and labels
    # -------------------------------------------------------
    mat = loadmat(structural_mat_file, struct_as_record=False, squeeze_me=True)

    # Adjust this according to your .mat structure:
    # Here we assume a struct MBM with field eig.beta_subject
    # shaped [nModes x nSubjects].
    if "MBM" in mat:
        beta_subject = mat["MBM"].eig.beta_subject
    else:
        # If you store it differently, adapt this line.
        beta_subject = mat["beta_subject"]

    if beta_subject.ndim != 2:
        raise ValueError(
            f"Expected 2D eigenmode matrix, got shape {beta_subject.shape}"
        )

    # [nModes x nSubjects] -> [nSubjects x nModes]
    X_struct = beta_subject.T

    df_group = pd.read_csv(group_file)
    if "group_numeric" not in df_group.columns:
        raise ValueError(
            "group_labels.csv must contain a 'group_numeric' column."
        )
    y = df_group["group_numeric"].values.astype(int)

    if X_struct.shape[0] != len(y):
        raise ValueError(
            "Number of rows in structural_modes does not match number "
            "of group labels.\n"
            f"  X_struct.shape[0] = {X_struct.shape[0]}\n"
            f"  len(y)            = {len(y)}"
        )

    X = pd.DataFrame(X_struct)

    # -------------------------------------------------------
    # 2. LOOCV with per-fold t-test feature selection
    # -------------------------------------------------------
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    )

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- t-test feature selection on training set ---
        group0 = X_train[y_train == 0]
        group1 = X_train[y_train == 1]

        p_vals = np.array(
            [
                ttest_ind(
                    group0.iloc[:, j],
                    group1.iloc[:, j],
                    equal_var=False,
                )[1]
                for j in range(X_train.shape[1])
            ]
        )

        feat_idx = np.where(p_vals < 0.05)[0]
        if len(feat_idx) == 0:
            # If nothing passes, just take the first feature.
            feat_idx = np.array([0])

        X_train_sel = X_train.iloc[:, feat_idx]
        X_test_sel = X_test.iloc[:, feat_idx]

        # --- Standardize ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # --- Neural network classifier ---
        model = Sequential(
            [
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.1),
                Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=70,
            callbacks=[early_stop],
            verbose=0,
        )

        prob = model.predict(X_test_scaled, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)

        y_prob.extend(prob.tolist())
        y_pred.extend(pred.tolist())
        y_true.extend(y_test.tolist())

    # -------------------------------------------------------
    # 3. Metrics
    # -------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)
    auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, random_state=SEED)

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc_val,
                "auc_ci_lower": auc_ci_lower,
                "auc_ci_upper": auc_ci_upper,
            }
        ]
    )
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"AUC:       {auc_val:.4f} (95% CI {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")

    # -------------------------------------------------------
    # 4. ROC Curve
    # -------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_val = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_val:.3f}; 95% CI {auc_ci_lower:.3f}-{auc_ci_upper:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Structural-mode classifier)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = results_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved ROC curve to: {roc_path}")


if __name__ == "__main__":
    main()
