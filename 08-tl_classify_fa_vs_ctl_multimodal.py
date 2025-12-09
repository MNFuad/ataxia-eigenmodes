#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tl_classify_fa_vs_ctl_multimodal.py

Purpose
-------
Transfer-learning style classifier using BOTH structural and fMRI
eigenmode weights to distinguish FA patients from healthy controls.

This script performs:
  • Load precomputed structural eigenmode weights per subject
  • Load precomputed fMRI mode weights per subject
  • Concatenate them into a multimodal feature vector
  • Train/test using Leave-One-Out Cross-Validation (LOOCV)
  • t-test feature selection inside each fold (on multimodal features)
  • Neural-network classifier with early stopping
  • Outputs performance metrics + ROC curve

Expected Input Files
--------------------
All inputs must reside under:

    repo_root/data/transfer_learning/multimodal/

1) structural_modes.mat
   - MATLAB file containing:
       MBM.eig.beta_subject   [nModes x nSubjects] OR [nSubjects x nModes]
   - This script assumes [nModes x nSubjects] and transposes to
     [nSubjects x nModes].

2) fmri_modes.mat
   - MATLAB file containing:
       MBAM.eig.betaCon_subject   [nModes x nSubjects] OR [nSubjects x nModes]
   - This script assumes [nModes x nSubjects] and transposes to
     [nSubjects x nModes].

3) group_labels.csv
   - CSV with at least:
       group_numeric
     where:
       0 = healthy control
       1 = FA
   - Optionally, may include:
       subject_id

Shape Requirements
------------------
Let:
  - nSub = number of subjects
  - nStructModes = # structural modes
  - nFmriModes   = # fMRI modes

We require that:
  - structural_modes has shape [nSub x nStructModes] after transpose
  - fmri_modes has shape [nSub x nFmriModes] after transpose
  - group_labels.csv has nSub rows

Directory Layout Example
------------------------
repo_root/
  data/
    transfer_learning/
      multimodal/
        structural_modes.mat
        fmri_modes.mat
        group_labels.csv

Outputs
-------
All outputs are saved under:

    repo_root/results/transfer_learning/
        tl_classify_fa_vs_ctl_multimodal/

Files generated:
  • metrics.csv
      - accuracy, precision, recall, f1, auc
  • roc_curve.png
      - ROC curve over the full LOOCV

Notes
-----
- This script does NOT generate any data by itself.
- It assumes consistent subject ordering across all inputs.
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


def main():
    # -------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "transfer_learning" / "multimodal"
    results_dir = (
        repo_root
        / "results"
        / "transfer_learning"
        / "tl_classify_fa_vs_ctl_multimodal"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    struct_mat_file = data_dir / "structural_modes.mat"
    fmri_mat_file = data_dir / "fmri_modes.mat"
    group_file = data_dir / "group_labels.csv"

    if not struct_mat_file.exists():
        raise FileNotFoundError(f"Missing input file: {struct_mat_file}")
    if not fmri_mat_file.exists():
        raise FileNotFoundError(f"Missing input file: {fmri_mat_file}")
    if not group_file.exists():
        raise FileNotFoundError(f"Missing group label file: {group_file}")

    # -------------------------------------------------------
    # 1. Load structural modes
    # -------------------------------------------------------
    mat_struct = loadmat(struct_mat_file, struct_as_record=False, squeeze_me=True)

    if "MBM" in mat_struct:
        beta_struct = mat_struct["MBM"].eig.beta_subject
    else:
        beta_struct = mat_struct["beta_subject"]

    if beta_struct.ndim != 2:
        raise ValueError(
            f"Expected 2D structural eigenmode matrix, got {beta_struct.shape}"
        )

    X_struct = beta_struct.T  # [nSubjects x nStructModes]

    # -------------------------------------------------------
    # 2. Load fMRI modes
    # -------------------------------------------------------
    mat_fmri = loadmat(fmri_mat_file, struct_as_record=False, squeeze_me=True)

    if "MBAM" in mat_fmri:
        beta_fmri = mat_fmri["MBAM"].eig.betaCon_subject
    else:
        beta_fmri = mat_fmri["betaCon_subject"]

    if beta_fmri.ndim != 2:
        raise ValueError(
            f"Expected 2D fMRI eigenmode matrix, got {beta_fmri.shape}"
        )

    X_fmri = beta_fmri.T  # [nSubjects x nFmriModes]

    # -------------------------------------------------------
    # 3. Load labels
    # -------------------------------------------------------
    df_group = pd.read_csv(group_file)
    if "group_numeric" not in df_group.columns:
        raise ValueError(
            "group_labels.csv must contain a 'group_numeric' column."
        )
    y = df_group["group_numeric"].values.astype(int)

    # Check consistency
    if X_struct.shape[0] != X_fmri.shape[0] or X_struct.shape[0] != len(y):
        raise ValueError(
            "Inconsistent subject counts among structural, fMRI, and label files.\n"
            f"  X_struct.shape[0] = {X_struct.shape[0]}\n"
            f"  X_fmri.shape[0]   = {X_fmri.shape[0]}\n"
            f"  len(y)            = {len(y)}"
        )

    # Concatenate features
    X_concat = np.hstack([X_struct, X_fmri])
    X = pd.DataFrame(X_concat)

    # -------------------------------------------------------
    # 4. LOOCV with per-fold t-test feature selection
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

        # --- t-test feature selection on training data ---
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
                Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.15),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.15),
                Dense(8, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.15),
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
            epochs=80,
            callbacks=[early_stop],
            verbose=0,
        )

        prob = model.predict(X_test_scaled, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)

        y_prob.extend(prob.tolist())
        y_pred.extend(pred.tolist())
        y_true.extend(y_test.tolist())

    # -------------------------------------------------------
    # 5. Metrics
    # -------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_prob)

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc_val,
            }
        ]
    )
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # -------------------------------------------------------
    # 6. ROC Curve
    # -------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_val = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multimodal classifier)")
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
