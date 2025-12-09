#!/usr/bin/env python
# coding: utf-8
"""
evaluate_fmri_reconstruction_from_predicted_modes.py
shorted as: eval_fmri_recon_from_pred_modes.py

Purpose
-------
Evaluate how well the predicted fMRI eigenmode weights (from
predict_fmri_eigenmodes_from_structure.py) can reconstruct the original
task fMRI activation maps (MBAM eigenmode-based maps).

Pipeline
--------
1. Load predicted fMRI eigenmode weights for each subject.
2. Load the MBAM eigenbasis (vertex-wise eigenmodes) and the
   "true" task activation maps (gammaCon_subject).
3. Align subjects between:
     - predictions (struct->func NN script)
     - MBAM (subject_ids.txt)
     - MBM (inputMaps_full_path.txt, for consistency)
4. Reconstruct predicted fMRI maps in vertex space:
       predicted_map = sum_k eigen_basis[:, k] * predicted_weight_k
5. Compute per-subject Pearson correlation between reconstructed and
   true maps.
6. Summarize reconstruction quality overall and by group (CTL vs FA).
7. Save metrics and simple plots under:

       results/functional_prediction/nn_struct2func/reconstruction/

Assumed repository layout
-------------------------
repo_root/
  data/
    mbm_example/
      inputMaps_full_path.txt
      ...
    mbam_example/
      MBAM/MBAM_data/
        MBAM_ckcpass_LR_30_164k_lh.mat
      subject_ids.txt
  results/
    functional_prediction/
      nn_struct2func/
        predictions/
          fmri_preds_all_modes_lh_ckcpass.npy
        ...

Notes
-----
From the dummy data, we assume:

    MBAM.eig.gammaCon_subject has shape [nVertices x nSubjects] = [100 x 30]

i.e. each column is one subject’s task map.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helper: extract subject ID from a full path (e.g., "sub-CTL01")
# ---------------------------------------------------------------------
def extract_subject_id(path_str: str) -> str:
    """
    Extract a subject ID of the form 'sub-XXX' from a full file path.

    Parameters
    ----------
    path_str : str
        Full path that contains 'sub-XXX' somewhere.

    Returns
    -------
    sid : str
        Subject ID string (e.g. 'sub-CTL01').

    Raises
    ------
    ValueError if no subject ID is found.
    """
    match = re.search(r"sub-[^/\\]+", path_str)
    if match is None:
        raise ValueError(f"Could not find subject ID in path: {path_str}")
    return match.group(0)


# ---------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # Basic configuration
    # -----------------------------------------------------------------
    hemisphere = "lh"    # matches dummy MBAM_ckcpass_LR_30_164k_lh.mat
    task = "ckcpass"     # matches dummy task

    # -----------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    data_dir = repo_root / "data"

    struct_dir = data_dir / "mbm_example"
    func_dir = data_dir / "mbam_example"

    # Predicted fMRI eigenmode weights from the NN prediction script
    pred_dir = (
        repo_root
        / "results"
        / "functional_prediction"
        / "nn_struct2func"
        / "predictions"
    )
    pred_file = pred_dir / f"fmri_preds_all_modes_{hemisphere}_{task}.npy"

    # MBAM data (.mat) with eigenbasis + true task maps
    fmri_mat_file = (
        func_dir / "MBAM" / "MBAM_data" / f"MBAM_{task}_LR_30_164k_{hemisphere}.mat"
    )

    # Subject ID files (for alignment)
    struct_list_file = struct_dir / "inputMaps_full_path.txt"
    func_id_file = func_dir / "subject_ids.txt"

    print("=== Paths ===")
    print(f"Repo root:          {repo_root}")
    print(f"Structural dir:     {struct_dir}")
    print(f"Functional dir:     {func_dir}")
    print(f"Predictions file:   {pred_file}")
    print(f"MBAM fMRI file:     {fmri_mat_file}")
    print("===============\n")

    # -----------------------------------------------------------------
    # Results directory
    # -----------------------------------------------------------------
    base_results_dir = (
        repo_root
        / "results"
        / "functional_prediction"
        / "nn_struct2func"
        / "reconstruction"
    )
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load predicted fMRI eigenmode weights
    # -----------------------------------------------------------------
    print("Loading predicted fMRI eigenmode weights...")
    if not pred_file.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_file}\n"
            "Make sure you ran predict_fmri_eigenmodes_from_structure.py first."
        )

    fmri_preds_all_modes = np.load(pred_file)  # shape: (nSubjects_pred, nModes_pred)
    n_sub_pred, n_modes_pred = fmri_preds_all_modes.shape
    print(
        f"  fmri_preds_all_modes shape: {fmri_preds_all_modes.shape} "
        "(nSubjects, nModes)"
    )

    # -----------------------------------------------------------------
    # Load MBAM fMRI data (eigenbasis + true activation maps)
    # -----------------------------------------------------------------
    print("\nLoading MBAM fMRI data...")
    if not fmri_mat_file.is_file():
        raise FileNotFoundError(f"MBAM fMRI file not found: {fmri_mat_file}")

    mbam_struct = loadmat(fmri_mat_file, struct_as_record=False, squeeze_me=True)["MBAM"]

    # Eigenbasis (vertex-wise eigenmodes)
    eigen_basis = mbam_struct.eig.eig  # [nVertices x nModes]
    if eigen_basis.ndim != 2:
        raise ValueError(
            f"Expected MBAM.eig.eig to be 2D, got shape {eigen_basis.shape}"
        )
    n_vertices, n_modes_eig = eigen_basis.shape
    print(f"  eigen_basis shape: {eigen_basis.shape} (nVertices, nModes)")

    # True maps in vertex space: gammaCon_subject
    # From your dummy data, this is 100 x 30 = [nVertices x nSubjects]
    gamma_true_full = mbam_struct.eig.gammaCon_subject
    print(f"  gammaCon_subject shape: {gamma_true_full.shape}")

    if gamma_true_full.ndim != 2:
        raise ValueError(
            f"Expected gammaCon_subject to be 2D, got shape {gamma_true_full.shape}"
        )
    if gamma_true_full.shape[0] != n_vertices:
        raise ValueError(
            f"gammaCon_subject first dimension (vertices) {gamma_true_full.shape[0]} "
            f"does not match eigen_basis vertices {n_vertices}."
        )

    # -----------------------------------------------------------------
    # Load structural & functional subject IDs for alignment
    # -----------------------------------------------------------------
    print("\nLoading subject IDs for alignment...")

    # Structural IDs: from inputMaps_full_path.txt
    if not struct_list_file.is_file():
        raise FileNotFoundError(f"Structural list file not found: {struct_list_file}")
    with open(struct_list_file, "r") as f:
        struct_map_paths = [line.strip() for line in f if line.strip()]
    struct_ids = [extract_subject_id(p) for p in struct_map_paths]

    # Functional IDs: from subject_ids.txt
    if not func_id_file.is_file():
        raise FileNotFoundError(f"Functional subject ID file not found: {func_id_file}")
    with open(func_id_file, "r") as f:
        func_ids = [line.strip() for line in f if line.strip()]

    print(f"  nStruct IDs: {len(struct_ids)}")
    print(f"  nFunc   IDs: {len(func_ids)}")

    if gamma_true_full.shape[1] != len(func_ids):
        print(
            "  [WARN] Number of columns in gammaCon_subject does not match "
            "functional subject IDs. Please verify consistency."
        )

    # -----------------------------------------------------------------
    # Align subjects via intersection of structural & functional IDs
    # -----------------------------------------------------------------
    struct_id_set = set(struct_ids)
    func_id_set = set(func_ids)
    common_ids = sorted(struct_id_set & func_id_set)

    if len(common_ids) == 0:
        raise RuntimeError(
            "No overlapping subject IDs between structural and functional datasets."
        )

    print(f"\nAligning to common IDs (n={len(common_ids)})")
    print(f"  First few common IDs: {common_ids[:5]}")

    # Index into func_ids to align gamma_true_full columns
    func_index = [func_ids.index(sid) for sid in common_ids]
    gamma_true_aligned = gamma_true_full[:, func_index]  # [nVertices x nCommon]

    # The prediction script also used sorted(common_ids), so rows of
    # fmri_preds_all_modes should correspond to this same ordering.
    if fmri_preds_all_modes.shape[0] != len(common_ids):
        raise ValueError(
            "Number of prediction subjects does not match number of common IDs.\n"
            f"  fmri_preds_all_modes.shape[0] = {fmri_preds_all_modes.shape[0]}\n"
            f"  len(common_ids)               = {len(common_ids)}"
        )

    # -----------------------------------------------------------------
    # Determine how many modes to use for reconstruction
    # -----------------------------------------------------------------
    n_modes_basis = eigen_basis.shape[1]
    nEigenmode_use = min(n_modes_pred, n_modes_basis)

    print(f"\nUsing nEigenmode_use = {nEigenmode_use} modes for reconstruction.")
    print(f"  (n_modes_pred={n_modes_pred}, n_modes_basis={n_modes_basis})")

    # -----------------------------------------------------------------
    # Reconstruct predicted fMRI maps in vertex space
    # -----------------------------------------------------------------
    print("\nReconstructing predicted fMRI activation maps...")
    # eigen_basis: [nVertices x nModes]
    # fmri_preds_all_modes: [nSubjects x nModes]
    fmri_recon_pred = (
        eigen_basis[:, :nEigenmode_use]
        @ fmri_preds_all_modes[:, :nEigenmode_use].T
    )  # [nVertices x nSubjects]

    if fmri_recon_pred.shape[1] != gamma_true_aligned.shape[1]:
        raise ValueError(
            "Mismatch between number of reconstructed subjects and true maps.\n"
            f"  fmri_recon_pred.shape[1]    = {fmri_recon_pred.shape[1]}\n"
            f"  gamma_true_aligned.shape[1] = {gamma_true_aligned.shape[1]}"
        )

    n_subjects = fmri_recon_pred.shape[1]

    # -----------------------------------------------------------------
    # Subject-wise correlation between predicted and true maps
    # -----------------------------------------------------------------
    print("\nComputing subject-wise reconstruction correlations...")
    correlations = np.zeros(n_subjects)
    for i in range(n_subjects):
        r, _ = pearsonr(fmri_recon_pred[:, i], gamma_true_aligned[:, i])
        correlations[i] = r

    # -----------------------------------------------------------------
    # Group labels from subject IDs (0 = Control, 1 = FA)
    # -----------------------------------------------------------------
    group_numeric = []
    for sid in common_ids:
        if sid.startswith("sub-CTL"):
            group_numeric.append(0)
        elif sid.startswith("sub-FA"):
            group_numeric.append(1)
        else:
            group_numeric.append(np.nan)
    group_numeric = np.array(group_numeric)
    group_label = pd.Series(group_numeric).map({0: "Control", 1: "FA"})

    # Filter out any NaN-labeled subjects if they exist (shouldn't for this demo)
    valid_mask = ~np.isnan(group_numeric)
    correlations = correlations[valid_mask]
    group_numeric = group_numeric[valid_mask]
    group_label = group_label[valid_mask]
    valid_ids = [sid for sid, keep in zip(common_ids, valid_mask) if keep]

    ctrl_corr = correlations[group_numeric == 0]
    fa_corr = correlations[group_numeric == 1]

    print("\n=== Reconstruction quality summary ===")
    print(
        f"Overall mean r = {correlations.mean():.3f} ± {correlations.std():.3f} "
        f"(range: {correlations.min():.3f}–{correlations.max():.3f})"
    )
    if len(ctrl_corr) > 0:
        print(
            f"Control mean r = {ctrl_corr.mean():.3f} ± {ctrl_corr.std():.3f} "
            f"(range: {ctrl_corr.min():.3f}–{ctrl_corr.max():.3f})"
        )
    if len(fa_corr) > 0:
        print(
            f"FA mean r      = {fa_corr.mean():.3f} ± {fa_corr.std():.3f} "
            f"(range: {fa_corr.min():.3f}–{fa_corr.max():.3f})"
        )

    if len(ctrl_corr) > 1 and len(fa_corr) > 1:
        t_stat, p_val = ttest_ind(ctrl_corr, fa_corr, equal_var=False)
        print(f"T-test (Control vs FA): t = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        print("Not enough subjects per group for a meaningful t-test.")

    # -----------------------------------------------------------------
    # Plots: histogram and boxplot
    # -----------------------------------------------------------------
    # Histogram
    plt.figure(figsize=(7, 5))
    if len(ctrl_corr) > 0:
        plt.hist(ctrl_corr, bins=10, alpha=0.6, label="Control", density=True)
    if len(fa_corr) > 0:
        plt.hist(fa_corr, bins=10, alpha=0.6, label="FA", density=True)
    plt.xlabel("Reconstruction correlation (Predicted vs True map)")
    plt.ylabel("Density")
    plt.title(f"Reconstruction accuracy distribution ({hemisphere}, {task})")
    plt.legend()
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()

    hist_path = base_results_dir / f"reconstruction_hist_{hemisphere}_{task}.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n📊 Histogram saved to: {hist_path}")

    # Boxplot
    plt.figure(figsize=(5, 5))
    data_for_box = []
    labels_for_box = []
    if len(ctrl_corr) > 0:
        data_for_box.append(ctrl_corr)
        labels_for_box.append("Control")
    if len(fa_corr) > 0:
        data_for_box.append(fa_corr)
        labels_for_box.append("FA")

    if data_for_box:
        plt.boxplot(data_for_box, labels=labels_for_box)
        plt.ylabel("Reconstruction correlation (r)")
        plt.title(f"Reconstruction performance by group ({hemisphere}, {task})")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        box_path = base_results_dir / f"reconstruction_box_{hemisphere}_{task}.png"
        plt.savefig(box_path, dpi=150)
        plt.close()
        print(f"📦 Boxplot saved to:     {box_path}")
    else:
        print("No valid group data to plot in boxplot.")

    # -----------------------------------------------------------------
    # Save numeric results
    # -----------------------------------------------------------------
    results_df = pd.DataFrame(
        {
            "SubjectID": valid_ids,
            "Group": group_label.values,
            "GroupNumeric": group_numeric.astype(int),
            "Correlation": correlations,
        }
    )

    csv_path = base_results_dir / f"reconstruction_corr_{hemisphere}_{task}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Reconstruction correlations saved to: {csv_path}")


if __name__ == "__main__":
    main()
