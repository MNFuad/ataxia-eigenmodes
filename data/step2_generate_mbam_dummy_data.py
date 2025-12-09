#!/usr/bin/env python
# coding: utf-8
"""
generate_mbam_dummy_data.py

Create a fully synthetic functional MBAM demo dataset under:

    repo_root/
      data/
        mbam_example/
          atlas/fsaverage/164k/lh/
            fsaverage_164k_midthickness-lh.vtk
            fsaverage_164k_cortex-lh_mask.txt
          derivatives/
            sub-CTL01/ses-1/func/...
            ...
            sub-FA30/ses-1/func/...
          MBAM/Results/
            sub-CTL01/MNINonLinear/tfMRI_ckcpass_ses-1.feat/design_mat.txt
            sub-CTL01/MNINonLinear/tfMRI_ckcpass_ses-1.feat/design_con.txt
            ...
          subject_ids.txt

Subjects:
    - 15 healthy controls:  sub-CTL01 ... sub-CTL15
    - 15 FA:                sub-FA16 ... sub-FA30

Group differences are encoded as opposite task-related activation
amplitudes between CTL and FA groups on a simple 10x10 grid surface.

Confounds are now non-zero, fMRIPrep-like time series so that the GLM
design matrix in eig_coef_func.m is not singular.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nibabel import gifti

# ------------------ CONFIG ------------------
REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = REPO_ROOT / "data" / "mbam_example"
ATLAS_LH   = DATA_ROOT / "atlas" / "fsaverage" / "164k" / "lh"
MBAM_ROOT  = DATA_ROOT / "MBAM"
DERIV_ROOT = DATA_ROOT / "derivatives"

TASK       = "ckcpass"
HEMISPHERE = "L"            # 'L' used in file names, 'lh' in atlas file names

# Simple toy surface: 10 x 10 grid
NX, NY     = 10, 10
N_VERTICES = NX * NY

# Number of time points per subject
N_TR       = 120

# Group sizes
N_CTL      = 15
N_FA       = 15
N_SUBJECTS = N_CTL + N_FA

np.random.seed(42)


def main():
    print(f"Creating dummy functional MBAM data under:\n  {DATA_ROOT}\n")
    ATLAS_LH.mkdir(parents=True, exist_ok=True)
    MBAM_ROOT.mkdir(parents=True, exist_ok=True)
    DERIV_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------ 1. Atlas: synthetic surface + mask ------------------
    vtk_path = ATLAS_LH / "fsaverage_164k_midthickness-lh.vtk"
    with open(vtk_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Synthetic cortical surface (rectangular grid)\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # vertices
        f.write(f"POINTS {N_VERTICES} float\n")
        for iy in range(NY):
            for ix in range(NX):
                x = ix / (NX - 1)
                y = iy / (NY - 1)
                z = 0.0
                f.write(f"{x} {y} {z}\n")

        # faces (two triangles per quad)
        n_cells = (NX - 1) * (NY - 1)
        n_tris  = n_cells * 2
        f.write(f"POLYGONS {n_tris} {n_tris * 4}\n")

        def vid(ix, iy):
            return iy * NX + ix

        for iy in range(NY - 1):
            for ix in range(NX - 1):
                v00 = vid(ix, iy)
                v10 = vid(ix + 1, iy)
                v01 = vid(ix, iy + 1)
                v11 = vid(ix + 1, iy + 1)
                f.write(f"3 {v00} {v10} {v11}\n")
                f.write(f"3 {v00} {v11} {v01}\n")

    print(f"  ✔ VTK surface written: {vtk_path}")

    mask = np.ones(N_VERTICES, dtype=int)
    mask_path = ATLAS_LH / "fsaverage_164k_cortex-lh_mask.txt"
    np.savetxt(mask_path, mask, fmt="%d")
    print(f"  ✔ Mask written:        {mask_path}\n")

    # ------------------ 2. Subject IDs (15 CTL + 15 FA) ------------------
    ctl_ids = [f"sub-CTL{i:02d}" for i in range(1, N_CTL + 1)]
    fa_ids  = [f"sub-FA{i:02d}"  for i in range(N_CTL + 1, N_CTL + N_FA + 1)]
    sub_ids = ctl_ids + fa_ids

    id_path = DATA_ROOT / "subject_ids.txt"
    id_path.write_text("\n".join(sub_ids))
    print(f"  ✔ subject_ids.txt written with {len(sub_ids)} IDs:")
    print(f"      {sub_ids[0]} ... {sub_ids[-1]}\n")

    # ------------------ 3. Template task regressor ------------------
    t = np.arange(N_TR)
    block = ((t // 20) % 2).astype(float)      # 0/1 blocks every 20 TRs
    task_reg = block - block.mean()            # demean
    X = task_reg[:, None]                      # (N_TR x 1)
    con = np.array([1.0])                      # single contrast

    # Spatial pattern (Gaussian bump) to modulate task activation
    x_coords = np.linspace(-1, 1, NX)
    y_coords = np.linspace(-1, 1, NY)
    xx, yy   = np.meshgrid(x_coords, y_coords)
    pattern  = np.exp(-(xx**2 + yy**2) / (2 * 0.4**2)).reshape(N_VERTICES)

    # ------------------ 4. Per-subject data ------------------
    for sid in sub_ids:
        print(f"  - Generating data for {sid}")

        # Group based on subject ID
        if sid.startswith("sub-CTL"):
            group = "CTL"
            beta  = 0.8    # amplitude of task response for controls
        elif sid.startswith("sub-FA"):
            group = "FA"
            beta  = -0.8   # opposite sign for FA to create group difference
        else:
            group = "UNK"
            beta  = 0.0

        subj_func_dir = DERIV_ROOT / sid / "ses-1" / "func"
        subj_func_dir.mkdir(parents=True, exist_ok=True)

        # signal: spatial pattern × task regressor × group-specific amplitude
        signal = np.outer(pattern, task_reg * beta)    # (N_VERT x N_TR)
        noise  = 0.3 * np.random.randn(N_VERTICES, N_TR)
        data   = (signal + noise).astype(np.float32)

        # ---- save .func.gii ----
        gii = gifti.GiftiImage(darrays=[gifti.GiftiDataArray(data)])
        func_name = (
            f"{sid}_ses-1_task-{TASK}_hemi-{HEMISPHERE}_space-fsaverage_bold.func.gii"
        )
        func_path = subj_func_dir / func_name
        nib.save(gii, str(func_path))
        print(f"      func.gii:       {func_path}")

        # ---- save confounds_timeseries.tsv ----
        # We create non-zero, fMRIPrep-like nuisance regressors.
        # This avoids singular design matrices in eig_coef_func.m.
        # Columns: motion, global/csf/WM, DVARS, FD, and a cosine regressor.
        motion = np.cumsum(0.05 * np.random.randn(N_TR, 6), axis=0)
        trans_x, trans_y, trans_z, rot_x, rot_y, rot_z = motion.T

        csf     = 0.1 * np.random.randn(N_TR)
        white_matter = 0.1 * np.random.randn(N_TR)
        global_signal = 0.1 * np.random.randn(N_TR)

        std_dvars = np.abs(0.05 * np.random.randn(N_TR))
        dvars     = np.abs(0.05 * np.random.randn(N_TR))

        # simple positive "FD" based on absolute framewise motion
        fd = np.zeros(N_TR)
        fd[1:] = np.abs(np.diff(trans_x)) + np.abs(np.diff(trans_y)) + np.abs(np.diff(trans_z))
        framewise_displacement = fd

        cosine00 = np.cos(2 * np.pi * t / N_TR)

        conf_df = pd.DataFrame({
            "trans_x": trans_x,
            "trans_y": trans_y,
            "trans_z": trans_z,
            "rot_x":   rot_x,
            "rot_y":   rot_y,
            "rot_z":   rot_z,
            "csf":     csf,
            "white_matter": white_matter,
            "global_signal": global_signal,
            "std_dvars": std_dvars,
            "dvars":    dvars,
            "framewise_displacement": framewise_displacement,
            "cosine00": cosine00,
        })

        conf_name = f"{sid}_ses-1_task-{TASK}_desc-confounds_timeseries.tsv"
        conf_path = subj_func_dir / conf_name
        conf_df.to_csv(conf_path, sep="\t", index=False)
        print(f"      confounds TSV: {conf_path}")

        # ---- save design_mat.txt and design_con.txt under MBAM/Results ----
        feat_dir = (
            MBAM_ROOT / "Results" / sid / "MNINonLinear" /
            f"tfMRI_{TASK}_ses-1.feat"
        )
        feat_dir.mkdir(parents=True, exist_ok=True)

        design_mat_path = feat_dir / "design_mat.txt"
        np.savetxt(design_mat_path, X, fmt="%.6f")
        print(f"      design_mat.txt: {design_mat_path}")

        design_con_path = feat_dir / "design_con.txt"
        np.savetxt(design_con_path, con, fmt="%.1f")
        print(f"      design_con.txt: {design_con_path}")

    print("\n✅ Dummy functional MBAM dataset created.")


if __name__ == "__main__":
    main()
