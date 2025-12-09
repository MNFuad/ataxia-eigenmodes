#!/usr/bin/env python
# coding: utf-8
"""
classify_sca1_vs_sca3_structural_modes.py

Purpose
-------
Neural network classifier distinguishing SCA1 from SCA3 using
structural MBM eigenmodes (no age matching / correction here,
matching the original SCA1-SCA3 script).

Pipeline
--------
1. Load MBM eigenmode weights: MBM.eig.beta_subject for SCA1 + SCA3.
2. Per-fold t-test feature selection (p < 0.05).
3. Train NN with 5-fold stratified CV.
4. Report accuracy / precision / recall / F1 / AUC.
5. Save ROC curve + metrics into:

       results/diagnosis/sca1_vs_sca3/

Expected data
-------------
repo_root/
  data/
    diagnosis/
      structural/
        SCA1_SCA3_rh.mat
          -> MBM.eig.beta_subject with rows: first group 0 (e.g. SCA1),
             then group 1 (e.g. SCA3).
      # No clinical age files required here.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


SEED = 41
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    # -----------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------
    hemisphere = "rh"
    n_features = 200
    n_splits = 5

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    data_dir = repo_root / "data" / "diagnosis"

    mbm_file = data_dir / "structural" / f"SCA1_SCA3_{hemisphere}.mat"
    results_dir = (
        repo_root / "results" / "diagnosis" / "sca1_vs_sca3"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paths ===")
    print(f"MBM file:    {mbm_file}")
    print(f"Results dir: {results_dir}")
    print("==============\n")

    # -----------------------------------------------------------------
    # Load MBM eigenmodes & group labels
    # -----------------------------------------------------------------
    if not mbm_file.is_file():
        raise FileNotFoundError(f"MBM file not found: {mbm_file}")

    mat = loadmat(mbm_file, struct_as_record=False, squeeze_me=True)
    mbm = mat["MBM"]
    beta_subject = mbm.eig.beta_subject
    X_all = beta_subject[:, :n_features]

    # Original script used:
    #   y_all = pd.Series([0]*88 + [1]*31, name='GroupNumeric') :contentReference[oaicite:11]{index=11}
    # i.e., first 88 subjects = group 0 (e.g. SCA1), next 31 = group 1 (e.g. SCA3).
    n_subjects_total = X_all.shape[0]
    n_group0 = 88
    n_group1 = 31

    if n_group0 + n_group1 != n_subjects_total:
        raise ValueError(
            "Group sizes (88 + 31) do not match total number of rows in MBM.\n"
            f"  n_subjects_total = {n_subjects_total}"
        )

    y_all = pd.Series(
        [0] * n_group0 + [1] * n_group1,
        name="GroupNumeric"
    )  # 0=SCA1, 1=SCA3

    print(f"Loaded MBM eigenmodes: X_all shape = {X_all.shape}")
    print(f"Group labels: {y_all.value_counts().to_dict()}")

    # -----------------------------------------------------------------
    # Cross-validated NN classification
    # -----------------------------------------------------------------
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_true, y_pred, y_probs = [], [], []

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(X_all, y_all), start=1
    ):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all.iloc[train_idx].values, y_all.iloc[test_idx].values

        # t-test feature selection (uncorrected p < 0.05) :contentReference[oaicite:12]{index=12}
        X_group1_train = X_train[y_train == 1]
        X_group0_train = X_train[y_train == 0]
        t_values, p_values = ttest_ind(
            X_group1_train, X_group0_train, equal_var=False
        )
        significant_features = np.where(p_values < 0.05)[0]

        if len(significant_features) == 0:
            print("  No significant features found; using all features.")
            significant_features = np.arange(X_all.shape[1])
        else:
            print(
                f"  Number of selected features = {len(significant_features)}"
            )

        X_train_sig = X_train[:, significant_features]
        X_test_sig  = X_test[:, significant_features]

        scaler = StandardScaler()
        X_train_sig = scaler.fit_transform(X_train_sig)
        X_test_sig  = scaler.transform(X_test_sig)

        # NN model
        model = Sequential(
            [
                Input(shape=(X_train_sig.shape[1],)),
                Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.1),
                Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.1),
                Dense(8, activation="relu", kernel_regularizer=l2(0.01)),
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
            X_train_sig,
            y_train,
            validation_data=(X_test_sig, y_test),
            epochs=75,
            callbacks=[early_stop],
            verbose=0,
        )

        probs = model.predict(X_test_sig, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

        y_probs.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(y_test.astype(int).tolist())

    # -----------------------------------------------------------------
    # Performance summary & ROC
    # -----------------------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_probs)

    print("\n5-Fold CV Classification Performance (SCA1 vs SCA3):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SCA1 vs SCA3 (Structural MBM modes)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = results_dir / f"roc_sca1_vs_sca3_{hemisphere}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_score,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = results_dir / f"metrics_sca1_vs_sca3_{hemisphere}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
