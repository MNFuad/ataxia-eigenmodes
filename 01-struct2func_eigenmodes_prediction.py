#!/usr/bin/env python
# coding: utf-8
"""
struct2func_eigenmodes_prediction.py

Purpose
-------
Predict task fMRI eigenmode contrasts (from MBAM) using structural
eigenmodes (from MBM) with a neural network regression model.

This script is intended for the FA-MUA GitHub demo and assumes the
following repository layout:

    repo_root/
      data/
        mbm_example/
          mbm_emp.mat                  # structural MBM results
          inputMaps_full_path.txt      # one anatomical map path per subject
          ...
        mbam_example/
          MBAM/MBAM_data/
            MBAM_ckcpass_LR_30_164k_lh.mat   # functional MBAM results
          subject_ids.txt              # one subject ID per line
          ...

Outputs are saved under:

    repo_root/
      results/
        functional_prediction/
          nn_struct2func/
            models/
            hyperparams/
            predictions/
            kt_dir/

High-level steps
----------------
1. Load structural eigenmode weights from MBM (`mbm_emp.mat`)
2. Load functional eigenmode contrasts from MBAM (`MBAM_...lh.mat`)
3. Align subjects via their IDs (intersection of structural + functional IDs)
4. For each fMRI mode:
    - Select top-k correlated structural modes
    - Use Keras Tuner to tune a small MLP
    - Run 5-fold CV and record R² and Pearson r
    - Train a final model on all data for that mode and save it
5. Save predictions and CV results for all modes
"""

import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner as kt


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------------------------------------------------------------
# Helper: extract subject ID from a full path
# -------------------------------------------------------------------------
def extract_subject_id(path_str: str) -> str:
    """
    Extract a subject ID of the form 'sub-XXX' from a full file path.
    Assumes BIDS-like naming, e.g.:

        .../sub-CTL01/surf/lh.thickness.fsaverage.mgh

    Returns
    -------
    sid : str
        The matched subject ID (e.g. 'sub-CTL01').

    Raises
    ------
    ValueError if no subject ID is found.
    """
    match = re.search(r"sub-[^/\\]+", path_str)
    if match is None:
        raise ValueError(f"Could not find subject ID in path: {path_str}")
    return match.group(0)


# -------------------------------------------------------------------------
# Model builder for Keras Tuner
# -------------------------------------------------------------------------
def build_model(hp: kt.engine.hyperparameters.HyperParameters, input_dim: int) -> Sequential:
    """
    Build a small MLP (multi-layer perceptron) model. Hyperparameters
    are tuned by Keras Tuner.

    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        Hyperparameter search space.
    input_dim : int
        Number of input features.

    Returns
    -------
    model : tf.keras.Model
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # First hidden layer
    model.add(
        Dense(
            units=hp.Int("units1", 16, 64, step=16),
            activation="relu",
            kernel_regularizer=l2(hp.Float("l2_1", 1e-3, 1e-1, sampling="log")),
        )
    )
    model.add(Dropout(hp.Float("dropout1", 0.1, 0.25, step=0.05)))

    # Second hidden layer
    model.add(
        Dense(
            units=hp.Int("units2", 8, 32, step=8),
            activation="relu",
            kernel_regularizer=l2(hp.Float("l2_2", 1e-3, 1e-1, sampling="log")),
        )
    )
    model.add(Dropout(hp.Float("dropout2", 0.1, 0.25, step=0.05)))

    # Single output (regression)
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float("lr", 1e-3, 1e-2, sampling="log")
        ),
        loss="mse",
    )
    return model


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def main():
    # ---------------------------------------------------------------------
    # Resolve repository paths
    # ---------------------------------------------------------------------
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent                   # script at repo root
    data_dir  = repo_root / "data"

    struct_dir = data_dir / "mbm_example"
    func_dir   = data_dir / "mbam_example"

    struct_mat_file = struct_dir / "mbm_emp.mat"
    func_mat_file   = func_dir / "MBAM" / "MBAM_data" / "MBAM_ckcpass_LR_30_164k_lh.mat"

    struct_list_file = struct_dir / "inputMaps_full_path.txt"
    func_id_file     = func_dir / "subject_ids.txt"

    print("=== Paths ===")
    print(f"Repo root:          {repo_root}")
    print(f"Structural MBM dir: {struct_dir}")
    print(f"Functional MBAM dir:{func_dir}")
    print(f"Structural MBM mat: {struct_mat_file}")
    print(f"Functional MBAM mat:{func_mat_file}")
    print("===============\n")

    # ---------------------------------------------------------------------
    # Results directory structure
    # ---------------------------------------------------------------------
    results_dir   = repo_root / "results" / "functional_prediction" / "nn_struct2func"
    models_dir    = results_dir / "models"
    hyperparam_dir = results_dir / "hyperparams"
    pred_dir      = results_dir / "predictions"
    kt_dir        = results_dir / "kt_dir"

    models_dir.mkdir(parents=True, exist_ok=True)
    hyperparam_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    kt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load structural MBM data (beta_subject)
    # ---------------------------------------------------------------------
    print("Loading structural MBM data...")

    if not struct_mat_file.is_file():
        raise FileNotFoundError(f"Structural MBM file not found: {struct_mat_file}")

    mbm = sio.loadmat(struct_mat_file, struct_as_record=False, squeeze_me=True)["MBM"]
    # Expect: MBM.eig.beta_subject is (nSubjects_struct x nModes_struct)
    struct_modes_full = mbm.eig.beta_subject

    if struct_modes_full.ndim != 2:
        raise ValueError(
            f"Expected MBM.eig.beta_subject to be 2D, got shape {struct_modes_full.shape}"
        )

    n_struct_subj, n_struct_modes = struct_modes_full.shape
    print(f"  Structural modes shape: {struct_modes_full.shape} "
          f"(nSubjects={n_struct_subj}, nModes={n_struct_modes})")

    # Subject order on structural side: from inputMaps_full_path.txt
    if not struct_list_file.is_file():
        raise FileNotFoundError(f"Structural map list not found: {struct_list_file}")

    with open(struct_list_file, "r") as f:
        struct_map_paths = [line.strip() for line in f if line.strip()]

    if len(struct_map_paths) != n_struct_subj:
        print("  [WARN] Number of anatomical map paths does not match "
              "number of rows in MBM.eig.beta_subject. Check consistency.")

    struct_ids = [extract_subject_id(p) for p in struct_map_paths]
    print(f"  Found {len(struct_ids)} structural subject IDs "
          f"(examples: {struct_ids[:3]} ...)\n")

    # ---------------------------------------------------------------------
    # Load functional MBAM data (betaCon_subject)
    # ---------------------------------------------------------------------
    print("Loading functional MBAM data...")

    if not func_mat_file.is_file():
        raise FileNotFoundError(f"Functional MBAM file not found: {func_mat_file}")

    mbam_mat = sio.loadmat(func_mat_file, struct_as_record=False, squeeze_me=True)["MBAM"]
    betaCon_subject = mbam_mat.eig.betaCon_subject

    # betaCon_subject might be:
    #   [nSubjects x nContrasts x nModes] or [nSubjects x nModes]
    if betaCon_subject.ndim == 3:
        n_func_subj, n_contrasts, n_func_modes = betaCon_subject.shape
        # Use the first contrast for the demo
        fmri_modes_full = betaCon_subject[:, 0, :]
    elif betaCon_subject.ndim == 2:
        n_func_subj, n_func_modes = betaCon_subject.shape
        fmri_modes_full = betaCon_subject
    else:
        raise ValueError(
            f"Unexpected betaCon_subject dimensions: {betaCon_subject.shape}"
        )

    print(f"  Functional modes shape: {fmri_modes_full.shape} "
          f"(nSubjects={n_func_subj}, nModes={n_func_modes})")

    # Replace any NaNs/Infs defensively
    if np.isnan(fmri_modes_full).any() or np.isinf(fmri_modes_full).any():
        print("  [WARN] fmri_modes_full contains NaN/Inf; replacing with zeros.")
        fmri_modes_full = np.nan_to_num(fmri_modes_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Functional subject IDs: from subject_ids.txt
    if not func_id_file.is_file():
        raise FileNotFoundError(f"Functional subject ID file not found: {func_id_file}")

    with open(func_id_file, "r") as f:
        func_ids = [line.strip() for line in f if line.strip()]

    if len(func_ids) != n_func_subj:
        print("  [WARN] Number of functional subject IDs does not match "
              "rows in MBAM.eig.betaCon_subject. Check consistency.")

    print(f"  Found {len(func_ids)} functional subject IDs "
          f"(examples: {func_ids[:3]} ...)\n")

    # ---------------------------------------------------------------------
    # Align subjects between structural and functional data
    # ---------------------------------------------------------------------
    print("Aligning structural and functional subjects...")

    struct_id_set = set(struct_ids)
    func_id_set   = set(func_ids)

    common_ids = sorted(struct_id_set & func_id_set)

    if len(common_ids) == 0:
        raise RuntimeError(
            "No overlapping subject IDs between structural and functional datasets."
        )

    print(f"  Number of overlapping subjects: {len(common_ids)}")
    print(f"  First few common IDs: {common_ids[:5]}")

    struct_index = [struct_ids.index(sid) for sid in common_ids]
    func_index   = [func_ids.index(sid)   for sid in common_ids]

    struct_modes = struct_modes_full[struct_index, :]
    fmri_modes   = fmri_modes_full[func_index, :]

    n_subjects = len(common_ids)
    print(f"  After alignment: struct_modes {struct_modes.shape}, "
          f"fmri_modes {fmri_modes.shape}\n")

    # ---------------------------------------------------------------------
    # Modelling parameters
    # ---------------------------------------------------------------------
    task       = "ckcpass"
    hemisphere = "lh"   # MBAM file is for left hemisphere
    resolution = "164k"

    # Number of fMRI modes to predict
    nEigenmode_choose = min(30, fmri_modes.shape[1])

    # Number of structural modes to consider as features
    n_struct_features = struct_modes.shape[1]
    top_k = min(25, n_struct_features)

    print("=== Modelling configuration ===")
    print(f"  Hemisphere            : {hemisphere}")
    print(f"  Task                  : {task}")
    print(f"  nEigenmode_choose     : {nEigenmode_choose}")
    print(f"  nStructFeatures avail : {n_struct_features}")
    print(f"  top_k                 : {top_k}")
    print("================================\n")

    struct_cols = [f"StructMode_{i+1}" for i in range(n_struct_features)]
    fmri_cols   = [f"FmriMode_{i+1}"   for i in range(fmri_modes.shape[1])]

    structure_modes_df = pd.DataFrame(struct_modes, columns=struct_cols)
    fmri_modes_df      = pd.DataFrame(fmri_modes,   columns=fmri_cols)

    # ---------------------------------------------------------------------
    # Correlation matrix: structural modes vs fMRI modes
    # ---------------------------------------------------------------------
    print("Computing correlation matrix between structural and fMRI modes...")

    n_structure = n_struct_features
    n_fmri      = fmri_modes_df.shape[1]

    correlation_matrix = np.zeros((n_structure, n_fmri), dtype=float)
    for i in range(n_structure):
        x = structure_modes_df.iloc[:, i]
        for j in range(n_fmri):
            y = fmri_modes_df.iloc[:, j]
            r, _ = pearsonr(x, y)
            correlation_matrix[i, j] = r

    print("  Correlation matrix shape:", correlation_matrix.shape, "\n")

    # ---------------------------------------------------------------------
    # Neural network prediction with Keras Tuner
    # ---------------------------------------------------------------------
    results = []
    fmri_preds_all_modes = np.zeros((n_subjects, nEigenmode_choose))

    for mode in range(nEigenmode_choose):
        print(f"\n🎯 Predicting fMRI mode {mode} / {nEigenmode_choose - 1}")

        # Feature selection: top_k structural modes most correlated with this fMRI mode
        top_k_indices = np.argsort(np.abs(correlation_matrix[:, mode]))[-top_k:][::-1]
        X_full = structure_modes_df.iloc[:, top_k_indices].copy()
        y_full = fmri_modes_df.iloc[:, mode]

        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X_full)

        # Hyperparameter tuning on full dataset
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, input_dim=X_full_scaled.shape[1]),
            objective="val_loss",
            max_trials=10,          # keep small for demo speed
            directory=str(kt_dir),
            project_name=f"mode_{mode}",
            overwrite=True,
        )

        tuner.search(
            X_full_scaled,
            y_full,
            validation_split=0.2,
            epochs=100,
            callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
            verbose=0,
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]

        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_r2, fold_corr = [], []
        y_pred_all = np.zeros_like(y_full.to_numpy(), dtype=float)

        for train_idx, test_idx in kf.split(X_full):
            X_train = X_full.iloc[train_idx]
            y_train = y_full.iloc[train_idx]
            X_test  = X_full.iloc[test_idx]
            y_test  = y_full.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            model = build_model(best_hps, input_dim=X_train_scaled.shape[1])
            model.fit(
                X_train_scaled,
                y_train,
                epochs=100,
                batch_size=len(X_train_scaled),
                validation_data=(X_test_scaled, y_test),
                callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
                verbose=0,
            )

            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            y_pred_all[test_idx] = y_pred

            r2 = r2_score(y_test, y_pred)
            r, _ = pearsonr(y_test, y_pred)
            fold_r2.append(r2)
            fold_corr.append(r)

        fmri_preds_all_modes[:, mode] = y_pred_all

        mean_r2, std_r2 = float(np.mean(fold_r2)), float(np.std(fold_r2))
        mean_corr, std_corr = float(np.mean(fold_corr)), float(np.std(fold_corr))

        print(f"   ✅ CV: R² = {mean_r2:.4f} ± {std_r2:.4f}, "
              f"Corr = {mean_corr:.4f} ± {std_corr:.4f}")

        results.append(
            {
                "mode": mode,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "mean_corr": mean_corr,
                "std_corr": std_corr,
                "units1": best_hps.get("units1"),
                "dropout1": best_hps.get("dropout1"),
                "l2_1": best_hps.get("l2_1"),
                "units2": best_hps.get("units2"),
                "dropout2": best_hps.get("dropout2"),
                "l2_2": best_hps.get("l2_2"),
                "lr": best_hps.get("lr"),
            }
        )

        # Train final model on full dataset for this mode and save
        best_model = build_model(best_hps, input_dim=X_full_scaled.shape[1])
        best_model.fit(
            X_full_scaled,
            y_full,
            epochs=50,
            batch_size=len(X_full_scaled),
            callbacks=[EarlyStopping(monitor="loss", patience=10)],
            verbose=0,
        )
        best_model.save(models_dir / f"mode_{mode}_model.h5")

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    out_csv = hyperparam_dir / f"cv_results_all_modes_{hemisphere}_{task}.csv"
    out_npy = pred_dir / f"fmri_preds_all_modes_{hemisphere}_{task}.npy"

    results_df.to_csv(out_csv, index=False)
    np.save(out_npy, fmri_preds_all_modes)

    print("\n💾 All models and predictions saved.")
    print(f"  CV results CSV : {out_csv}")
    print(f"  Predictions NPY: {out_npy}")

    print(f"\n📈 Overall mean R² across modes: "
          f"{results_df['mean_r2'].mean():.4f} ± {results_df['mean_r2'].std():.4f}")
    print(f"📈 Overall mean Corr across modes: "
          f"{results_df['mean_corr'].mean():.4f} ± {results_df['mean_corr'].std():.4f}")


if __name__ == "__main__":
    main()
