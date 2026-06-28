#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tl_classify_fa_vs_ctl_using_fmri_modes.py

Purpose
-------
Transfer-learning style classifier using ONLY fMRI geometric eigenmode
weights to distinguish FA patients from healthy controls.

This script performs:
  • Load precomputed fMRI eigenmode weights per subject
  • Train/test using Leave-One-Out CV
  • t-test feature selection inside each fold
  • Neural-network classifier with early stopping
  • Outputs performance metrics + ROC curve

Expected Input Files (must be placed under data/transfer_learning/fmri/)
------------------------------------------------------------------------
1. fmri_modes.mat  (MATLAB file containing MBAM.eig.betaCon_subject)
    Required variable:
        MBAM.eig.betaCon_subject   [nModes x nSubjects]

2. group_labels.csv
    Columns:
        subject_id, group_numeric
    where group_numeric = 0 for control, 1 for FA

Directory Structure Example
---------------------------
repo_root/
  data/
    transfer_learning/
      fmri/
        fmri_modes.mat
        group_labels.csv

Outputs
-------
Saved under:
    results/transfer_learning/tl_classify_fa_vs_ctl_using_fmri_modes/

Files generated:
    metrics.csv
    roc_curve.png

Compatibility
-------------
Python 3.9+
TensorFlow ≥ 2.9
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import LeaveOneOut
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

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "transfer_learning" / "fmri"
    results_dir = repo_root / "results" / "transfer_learning" / \
        "tl_classify_fa_vs_ctl_using_fmri_modes"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------
    fmri_mat_file = data_dir / "fmri_modes.mat"
    group_file = data_dir / "group_labels.csv"

    if not fmri_mat_file.exists():
        raise FileNotFoundError(f"Missing input file: {fmri_mat_file}")
    if not group_file.exists():
        raise FileNotFoundError(f"Missing group label file: {group_file}")

    mat = loadmat(fmri_mat_file, struct_as_record=False, squeeze_me=True)
    fmri_modes = mat["MBAM"].eig.betaCon_subject.T  # shape: subjects x modes

    df_group = pd.read_csv(group_file)
    y = df_group["group_numeric"].values.astype(int)

    X = pd.DataFrame(fmri_modes)

    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    # -------------------------------------------------------
    # 2. LOOCV
    # -------------------------------------------------------
    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # t-test feature selection
        group0 = X_train[y_train == 0]
        group1 = X_train[y_train == 1]

        p_vals = np.array([
            ttest_ind(group0[col], group1[col], equal_var=False)[1]
            for col in X_train.columns
        ])
        feat_idx = np.where(p_vals < 0.05)[0]
        if len(feat_idx) == 0:
            feat_idx = np.array([0])

        X_train_sel = X_train.iloc[:, feat_idx]
        X_test_sel = X_test.iloc[:, feat_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # Neural network
        model = Sequential([
            Input(shape=(X_train_scaled.shape[1],)),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=70,
            callbacks=[early_stop],
            verbose=0
        )

        probs = model.predict(X_test_scaled).flatten()
        preds = (probs > 0.5).astype(int)

        y_prob.extend(probs)
        y_pred.extend(preds)
        y_true.extend(y_test)

    # -------------------------------------------------------
    # 3. Metrics
    # -------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)
    auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(y_true, y_prob, n_bootstraps=2000, random_state=SEED)

    metrics = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_val,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper
    }])
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    print(f"AUC:       {auc_val:.4f} (95% CI {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")

    # -------------------------------------------------------
    # 4. ROC Curve
    # -------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_val = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_val:.3f}; 95% CI {auc_ci_lower:.3f}-{auc_ci_upper:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (fMRI mode classifier)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
