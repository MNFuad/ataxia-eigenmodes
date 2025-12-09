#!/usr/bin/env python
# coding: utf-8
"""
classify_fa_vs_sca1_structural_modes.py

Purpose
-------
Age-range-matched neural network classifier distinguishing
Friedreich's ataxia (FA) from spinocerebellar ataxia (SCA1) using structural MBM eigenmodes.

Pipeline
--------
1. Load MBM eigenmode weights: MBM.eig.beta_subject for FA + SCA1.
2. Load age for both groups from Excel clinical files.
3. Restrict to overlapping age range between groups.
4. t-test feature selection (p < 0.05) on training folds.
5. Train NN with 5-fold stratified CV.
6. Report accuracy / precision / recall / F1 / AUC.
7. Save ROC curve + metrics into:

       results/diagnosis/fa_vs_sca1/

Expected data
-------------
repo_root/
  data/
    diagnosis/
      structural/
        SCA1_FA_rh.mat
          -> MBM.eig.beta_subject with rows: FA (first), then SCA1
             (or vice versa; see note in CONFIG)
      clinical/
        SCA1_ClinicalData.xlsx   (Age column)
        FRDA_ClinicalData.xlsx   (Age column)

If your order in MBM file is [FA ; SCA1], set ORDER_FA_FIRST = True below.
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

    # IMPORTANT: set this according to how SCA1_FA_rh.mat is ordered.
    # Original script assumed [FA ; SCA1] split using len(age_fa) first. :contentReference[oaicite:7]{index=7}
    ORDER_FA_FIRST = True

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    data_dir = repo_root / "data" / "diagnosis"

    mbm_file = data_dir / "structural" / f"SCA1_FA_{hemisphere}.mat"
    age_sca1_file = data_dir / "clinical" / "SCA1_ClinicalData.xlsx"
    age_fa_file   = data_dir / "clinical" / "FRDA_ClinicalData.xlsx"

    results_dir = (
        repo_root / "results" / "diagnosis" / "fa_vs_sca1"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paths ===")
    print(f"MBM file:        {mbm_file}")
    print(f"SCA1 age file:   {age_sca1_file}")
    print(f"FA age file:     {age_fa_file}")
    print(f"Results dir:     {results_dir}")
    print("==============\n")

    # -----------------------------------------------------------------
    # Load MBM eigenmodes
    # -----------------------------------------------------------------
    if not mbm_file.is_file():
        raise FileNotFoundError(f"MBM file not found: {mbm_file}")

    mat = loadmat(mbm_file, struct_as_record=False, squeeze_me=True)
    mbm = mat["MBM"]
    beta_subject = mbm.eig.beta_subject
    X_all = beta_subject[:, :n_features]
    n_subjects_total = X_all.shape[0]

    print(f"Loaded MBM eigenmodes: X_all shape = {X_all.shape}")

    # -----------------------------------------------------------------
    # Load ages
    # -----------------------------------------------------------------
    if not age_sca1_file.is_file():
        raise FileNotFoundError(f"SCA1 age file not found: {age_sca1_file}")
    if not age_fa_file.is_file():
        raise FileNotFoundError(f"FA age file not found: {age_fa_file}")

    age_sca1 = pd.read_excel(age_sca1_file)["Age"].values
    age_fa   = pd.read_excel(age_fa_file)["Age"].values

    n_sca1 = len(age_sca1)
    n_fa   = len(age_fa)

    if n_sca1 + n_fa != n_subjects_total:
        raise ValueError(
            "Mismatch between MBM subjects and age arrays.\n"
            f"  n_subjects_total = {n_subjects_total}\n"
            f"  len(age_sca1)    = {n_sca1}\n"
            f"  len(age_fa)      = {n_fa}"
        )

    print(f"SCA1 subjects: {n_sca1}, FA subjects: {n_fa}")
    print(f"Mean age SCA1: {age_sca1.mean():.2f}, FA: {age_fa.mean():.2f}")

    # -----------------------------------------------------------------
    # Age matching (overlapping range)
    # -----------------------------------------------------------------
    A = age_sca1.mean()
    B = age_fa.mean()
    age_min, age_max = min(A, B), max(A, B)

    sca1_idx_keep = np.where((age_sca1 >= age_min) & (age_sca1 <= age_max))[0]
    fa_idx_keep   = np.where((age_fa   >= age_min) & (age_fa   <= age_max))[0]

    if ORDER_FA_FIRST:
        X_fa_matched   = X_all[:n_fa][fa_idx_keep]
        X_sca1_matched = X_all[n_fa:][sca1_idx_keep]
    else:
        X_sca1_matched = X_all[:n_sca1][sca1_idx_keep]
        X_fa_matched   = X_all[n_sca1:][fa_idx_keep]

    X_matched = np.vstack([X_fa_matched, X_sca1_matched])
    y_matched = np.array(
        [1] * len(fa_idx_keep) + [0] * len(sca1_idx_keep)
    )  # 1=FA, 0=SCA1 (same as original y_matched) :contentReference[oaicite:8]{index=8}

    print("\nAfter age matching:")
    print(f"  SCA1 kept: {len(sca1_idx_keep)}, FA kept: {len(fa_idx_keep)}")
    print(
        f"  SCA1 mean age: {age_sca1[sca1_idx_keep].mean():.2f}, "
        f"FA mean age: {age_fa[fa_idx_keep].mean():.2f}"
    )
    print(f"  X_matched shape: {X_matched.shape}")

    # -----------------------------------------------------------------
    # Cross-validated NN classification
    # -----------------------------------------------------------------
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_true, y_pred, y_probs = [], [], []

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(X_matched, y_matched), start=1
    ):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X_matched[train_idx], X_matched[test_idx]
        y_train, y_test = y_matched[train_idx], y_matched[test_idx]

        # t-test feature selection :contentReference[oaicite:9]{index=9}
        X_fa_train   = X_train[y_train == 1]
        X_sca1_train = X_train[y_train == 0]
        t_values, p_values = ttest_ind(X_fa_train, X_sca1_train, equal_var=False)
        significant_features = np.where(p_values < 0.05)[0]

        if len(significant_features) == 0:
            print("  No significant features found, using all.")
            significant_features = np.arange(X_matched.shape[1])
        else:
            print(f"  Selected {len(significant_features)} significant features.")

        X_train_sig = X_train[:, significant_features]
        X_test_sig  = X_test[:, significant_features]

        scaler = StandardScaler()
        X_train_sig = scaler.fit_transform(X_train_sig)
        X_test_sig  = scaler.transform(X_test_sig)

        # NN model
        model = Sequential(
            [
                Input(shape=(X_train_sig.shape[1],)),
                Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.15),
                Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.15),
                Dense(8, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.15),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.02),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        model.fit(
            X_train_sig,
            y_train,
            validation_data=(X_test_sig, y_test),
            epochs=100,
            callbacks=[early_stop],
            verbose=0,
        )

        probs = model.predict(X_test_sig, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

        y_probs.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(y_test.astype(int).tolist())

    # -----------------------------------------------------------------
    # Performance & ROC
    # -----------------------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_probs)

    print("\n========== 5-Fold CV Performance (FA vs SCA1, age-matched) ==========")
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
    plt.title("FA vs SCA1 (Age-range-matched)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = results_dir / f"roc_fa_vs_sca1_{hemisphere}.png"
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
    metrics_path = results_dir / f"metrics_fa_vs_sca1_{hemisphere}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
