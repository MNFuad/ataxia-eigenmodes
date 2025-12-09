import numpy as np
import os
from pathlib import Path
import nibabel as nib
from nibabel.freesurfer.mghformat import MGHImage

# ------------------ CONFIG ------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "mbm_example"
DERIV_DIR = OUTPUT_DIR / "derivatives_freesurfer"

# subjects per group
N_PER_GROUP = 15              # 12 CTL + 12 FA
N_SUBJECTS = 2 * N_PER_GROUP  # 24 total

# simple rectangular grid surface: NX * NY vertices
NX, NY = 10, 10
N_VERTICES = NX * NY

print(f"Using synthetic surface with {N_VERTICES} vertices ({NX} x {NY}).")
print(f"Subjects: {N_PER_GROUP} controls + {N_PER_GROUP} FA = {N_SUBJECTS} total.\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ 1. VTK surface ------------------
vtk_path = OUTPUT_DIR / "fsaverage_164k_midthickness-lh.vtk"
with open(vtk_path, "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Synthetic cortical surface (rectangular grid)\n")
    f.write("ASCII\n")
    f.write("DATASET POLYDATA\n")

    # POINTS
    f.write(f"POINTS {N_VERTICES} float\n")
    for iy in range(NY):
        for ix in range(NX):
            x = ix / (NX - 1)
            y = iy / (NY - 1)
            z = 0.0
            f.write(f"{x} {y} {z}\n")

    # POLYGONS (triangles)
    n_cells = (NX - 1) * (NY - 1)
    n_tris = n_cells * 2
    f.write(f"POLYGONS {n_tris} {n_tris * 4}\n")

    def vid(ix, iy):
        return iy * NX + ix

    for iy in range(NY - 1):
        for ix in range(NX - 1):
            v00 = vid(ix, iy)
            v10 = vid(ix + 1, iy)
            v01 = vid(ix, iy + 1)
            v11 = vid(ix + 1, iy + 1)
            # two triangles per quad
            f.write(f"3 {v00} {v10} {v11}\n")
            f.write(f"3 {v00} {v11} {v01}\n")

print(f"✅ VTK surface: {vtk_path}")

# ------------------ 2. Mask (all vertices included) ------------------
mask = np.ones(N_VERTICES, dtype=int)
mask_path = OUTPUT_DIR / "fsaverage_164k_cortex-lh_mask.txt"
np.savetxt(mask_path, mask, fmt="%d")
print(f"✅ Mask: {mask_path}")

# ------------------ 3. Group spatial pattern ------------------
# Create a smooth spatial pattern that varies across the grid
x_coords = np.linspace(-1, 1, NX)
y_coords = np.linspace(-1, 1, NY)
xx, yy = np.meshgrid(x_coords, y_coords)
pattern = np.exp(- (xx**2 + yy**2) / (2 * 0.3**2))  # Gaussian bump in the center
pattern = pattern.reshape(N_VERTICES)

# effect size for FA group
EFFECT_SIZE = 1.0  # increase this for stronger group differences

NOISE_SD    = 0.3     # non-zero noise

# ------------------ 4. Generate maps (.mgh) ------------------
DERIV_DIR.mkdir(parents=True, exist_ok=True)
input_paths = []

for i in range(N_SUBJECTS):
    if i < N_PER_GROUP:
        group = "CTL"
        group_label = 0
    else:
        group = "FA"
        group_label = 1

    sid = f"sub-{group}{i+1:02d}"
    surf_dir = DERIV_DIR / sid / "surf"
    surf_dir.mkdir(parents=True, exist_ok=True)

    # baseline noise
    base = np.random.normal(0.0, NOISE_SD, N_VERTICES)

    # group effect: only FA has an added pattern
    if group_label == 0: # CTL
        data = base
    else: # FA
        data = base + EFFECT_SIZE * pattern

    # shift to positive "thickness-like" range
    data = 2.5 + data
    data = data.astype(np.float32)

    # MGH shape (N_VERTICES, 1, 1)
    data_3d = data.reshape(N_VERTICES, 1, 1)
    img = MGHImage(data_3d, affine=np.eye(4, dtype=np.float32))

    out_path = surf_dir / "lh.thickness.fsaverage.mgh"
    nib.save(img, str(out_path))

    input_paths.append(str(out_path.resolve()).replace("\\", "/"))
    print(f"  - Map for {sid} saved at {out_path}")

# ------------------ 5. inputMaps_full_path.txt ------------------
input_list_path = OUTPUT_DIR / "inputMaps_full_path.txt"
with open(input_list_path, "w") as f:
    f.write("\n".join(input_paths))
print(f"\n✅ inputMaps_full_path.txt: {input_list_path}")

# ------------------ 6. G_two_sample.txt (dummy-coded) ------------------
G = np.zeros((N_SUBJECTS, 2), dtype=int)
G[:N_PER_GROUP, 0] = 1      # controls
G[N_PER_GROUP:, 1] = 1      # FA

G_path = OUTPUT_DIR / "G_two_sample.txt"
np.savetxt(G_path, G, fmt="%d")
print(f"✅ G_two_sample.txt: {G_path}")

print("\nAll synthetic MBM demo files generated.\n")
