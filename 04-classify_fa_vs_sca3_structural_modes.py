#!/usr/bin/env python
# coding: utf-8
"""
classify_fa_vs_sca3_structural_modes.py

Purpose
-------
Age-range-matched neural network classifier distinguishing
Friedreich's ataxia (FA) from SCA3 using structural MBM eigenmodes.

This script:
  1. Loads MBM eigenmode weights: MBM.eig.beta_subject
     for SCA3 + FA subjects.
  2. Loads age for both groups from Excel clinical files.
  3. Restricts to an overlapping age range between groups.
  4. Performs per-fold t-test feature selection (p < 0.05).
  5. Trains a small neural network with 5-fold stratified CV.
  6. Reports accuracy / precision / recall / F1 / AUC.
  7. Saves ROC curve and metrics under:

       results/diagnosis/fa_vs_sca3/

Expected data (you must provide)
--------------------------------
repo_root/
  data/
    diagnosis/
      structural/
        SCA3_lh.mat
          -> contains struct MBM struct with field MBM.eig.beta_subject
             (rows: first SCA3 subjects, then FA subjects)
      clinical/
        SCA3_ClinicalData.xlsx    (column: 'Age')
        FRDA_ClinicalData.xlsx    (column: 'Age')

If your filenames differ, edit the DATA PATHS section below.
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


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 41
random.seed(SEED)
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
    # -----------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------
    hemisphere = "lh"
    n_features = 200       # first N MBM eigenmodes
    n_splits = 5           # stratified CV folds

    # Resolve repo & data paths
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    data_dir = repo_root / "data" / "diagnosis"

    mbm_file = data_dir / "structural" / f"SCA3_{hemisphere}.mat"
    age_sca3_file = data_dir / "clinical" / "SCA3_ClinicalData.xlsx"
    age_fa_file   = data_dir / "clinical" / "FRDA_ClinicalData.xlsx"

    results_dir = (
        repo_root / "results" / "diagnosis" / "fa_vs_sca3"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paths ===")
    print(f"MBM file:       {mbm_file}")
    print(f"SCA3 age file:  {age_sca3_file}")
    print(f"FA age file:    {age_fa_file}")
    print(f"Results dir:    {results_dir}")
    print("==============\n")

    # -----------------------------------------------------------------
    # Load MBM eigenmodes (SCA3 + FA)
    # -----------------------------------------------------------------
    if not mbm_file.is_file():
        raise FileNotFoundError(f"MBM file not found: {mbm_file}")

    mat = loadmat(mbm_file, struct_as_record=False, squeeze_me=True)
    mbm = mat["MBM"]
    beta_subject = mbm.eig.beta_subject  # shape: [nSubjects_total x nModes]
    X_all = beta_subject[:, :n_features]
    n_subjects_total = X_all.shape[0]

    print(f"Loaded MBM eigenmodes: X_all shape = {X_all.shape}")

    # -----------------------------------------------------------------
    # Load ages (SCA3 then FA) and match to MBM subjects
    # -----------------------------------------------------------------
    if not age_sca3_file.is_file():
        raise FileNotFoundError(f"SCA3 age file not found: {age_sca3_file}")
    if not age_fa_file.is_file():
        raise FileNotFoundError(f"FA age file not found: {age_fa_file}")

    age_sca3 = pd.read_excel(age_sca3_file)["Age"].values
    age_fa   = pd.read_excel(age_fa_file)["Age"].values

    n_sca3 = len(age_sca3)
    n_fa   = len(age_fa)

    if n_sca3 + n_fa != n_subjects_total:
        raise ValueError(
            "Mismatch between MBM subjects and age arrays.\n"
            f"  n_subjects_total = {n_subjects_total}\n"
            f"  len(age_sca3)    = {n_sca3}\n"
            f"  len(age_fa)      = {n_fa}"
        )

    print(f"SCA3 subjects: {n_sca3}, FA subjects: {n_fa}")
    print(f"Mean age SCA3: {age_sca3.mean():.2f}, FA: {age_fa.mean():.2f}")

    # -----------------------------------------------------------------
    # Age-range matching
    # -----------------------------------------------------------------
    A = age_sca3.mean()
    B = age_fa.mean()
    age_min, age_max = min(A, B) - 5, max(A, B)  # same as original script

    sca3_idx_keep = np.where((age_sca3 >= age_min) & (age_sca3 <= age_max))[0]
    fa_idx_keep   = np.where((age_fa   >= age_min) & (age_fa   <= age_max))[0]

    X_sca3_matched = X_all[:n_sca3][sca3_idx_keep]
    X_fa_matched   = X_all[n_sca3:][fa_idx_keep]

    X_matched = np.vstack([X_sca3_matched, X_fa_matched])
    y_matched = np.array(
        [0] * len(sca3_idx_keep) + [1] * len(fa_idx_keep)
    )  # 0=SCA3, 1=FA

    print("\nAfter age-range matching:")
    print(f"  SCA3 kept: {len(sca3_idx_keep)}, FA kept: {len(fa_idx_keep)}")
    print(
        f"  SCA3 mean age: {age_sca3[sca3_idx_keep].mean():.2f}, "
        f"FA mean age: {age_fa[fa_idx_keep].mean():.2f}"
    )
    print(f"  X_matched shape: {X_matched.shape}")

    # -----------------------------------------------------------------
    # Cross-validated NN classification with per-fold t-test feature selection
    # -----------------------------------------------------------------
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_true, y_pred, y_probs = [], [], []

    early_stop = EarlyStopping(
        monitor="val_accuracy",
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

        # t-test feature selection (FA vs SCA3) :contentReference[oaicite:4]{index=4}
        X_fa_train   = X_train[y_train == 1]
        X_sca3_train = X_train[y_train == 0]
        t_values, p_values = ttest_ind(X_fa_train, X_sca3_train, equal_var=False)
        significant_features = np.where(p_values < 0.05)[0]

        if len(significant_features) == 0:
            print("  No significant features found, using all.")
            significant_features = np.arange(X_matched.shape[1])
        else:
            print(f"  Selected {len(significant_features)} significant features.")

        # Standardize
        X_train_sig = X_train[:, significant_features]
        X_test_sig  = X_test[:, significant_features]

        scaler = StandardScaler()
        X_train_sig = scaler.fit_transform(X_train_sig)
        X_test_sig  = scaler.transform(X_test_sig)

        # NN model :contentReference[oaicite:5]{index=5}
        model = Sequential(
            [
                Input(shape=(X_train_sig.shape[1],)),
                Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.1),
                Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.1),
                Dense(8, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.1),
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

        # Prediction
        probs = model.predict(X_test_sig, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

        y_probs.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(y_test.astype(int).tolist())

    # -----------------------------------------------------------------
    # Performance summary
    # -----------------------------------------------------------------
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_probs)
    auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(y_true, y_probs, n_bootstraps=2000, random_state=SEED)

    print("\n========== 5-Fold CV Performance (Age-range-matched) ==========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f} (95% CI {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f}; 95% CI {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FA vs SCA3 (Age-range-matched)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = results_dir / f"roc_fa_vs_sca3_{hemisphere}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_score,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = results_dir / f"metrics_fa_vs_sca3_{hemisphere}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
