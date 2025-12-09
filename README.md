# ataxia-eigenmodes: Structural & Functional Brain Geometry for Rare Hereditary Ataxias
**Geometric Eigenmodes • FA / SCA Diagnosis • Structure-to-Function Prediction**

[![Python](https://img.shields.io/badge/Python-3.8–3.10-blue.svg)]()
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020+-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%202-blue.svg)]()

This repository accompanies the work:

> **Structural MRI based Brain Geometry Signatures for Diagnosing Rare Hereditary Ataxias and Predicting Functional Profiles**

It implements a complete pipeline for **geometric eigenmode analysis** of structural and functional MRI in Friedreich's Ataxia (FA) and spinocerebellar ataxias (SCA).

---

# 📑 Table of Contents

1. [Overview](#-overview)  
2. [Repository Structure](#-repository-structure)  
3. [Installation](#-installation)  
   - [MATLAB](#matlab)  
   - [Python](#python)  
4. [Workflow Overview](#-workflow-overview)  
5. [Step-by-Step Pipeline](#step-by-step-pipeline)  
6. [Expected Data Directory](#-expected-data-directory)  
7. [Dummy vs User-Provided Data](#dummy-vs-user-provided-data)  
8. [Scientific Background & Citations](#-scientific-background--citations)  
9. [How to Cite](#-how-to-cite)  

---

# 🔍 Overview

This repo provides a complete framework to:

- Generate **dummy structural and functional MRI datasets**  
- Run **MBM** (structural Mode-Based Morphometry) in MATLAB  
- Run **MBAM** (Mode-Based Activation Mapping) for task fMRI in MATLAB  
- Train deep-learning models to:
  - predict fMRI eigenmode weights from structural eigenmodes  
  - reconstruct full fMRI activation maps  
  - classify FA vs SCA1 vs SCA3  
  - perform transfer learning across cohorts  

Dummy data support only the MBM/MBAM demos and structure→function prediction.  
Real diagnosis and transfer-learning require **user-provided datasets**.

---

# 🗂 Repository Structure

```text
repo_root/
├── 01-struct2func_eigenmodes_prediction.py
├── 02-eval_fmri_recon_from_pred_modes.py
├── 03-classify_fa_vs_sca1_structural_modes.py
├── 04-classify_fa_vs_sca3_structural_modes.py
├── 05-classify_sca1_vs_sca3_structural_modes.py
├── 06-tl_classify_fa_vs_ctl_using_fmri_modes.py
├── 07-tl_classify_fa_vs_ctl_using_structural_modes.py
├── 08-tl_classify_fa_vs_ctl_multimodal.py
├── data/
│   ├── mbm_example/                 
│   ├── mbam_example/                
│   ├── step1_generate_mbm_dummy_data.py
│   └── step2_generate_mbam_dummy_data.py
├── matlab/
│   ├── mbm/
│   │   ├── eig_coef_func.m
│   │   ├── step1_structural_mbm.m
│   │   └── step2_functional_mbam.m
│   └── tools/
│       ├── MBM-main/               
│       └── cifti-matlab/           
└── docs/
    ├── workflow_overview.png       
    └── eigenmodes_basis.png        
```

---

# ⚙️ Installation

## MATLAB

Requires external toolboxes:

- **MBM-main**
- **MBAM**
- **cifti-matlab**

Add to MATLAB path:

```matlab
addpath('matlab/tools/MBM-main');
addpath('matlab/tools/cifti-matlab');
```

---

## Python

### Conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate ataxia-eigenmodes
```

### Or install manually:

```bash
pip install -r requirements.txt
```

---

# 🔄 Workflow Overview

```
        ┌─────────────────────────────────────────┐
        │ 1. Generate Dummy Data (Python)         │
        └─────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │ 2. Run MBM (Structure) in MATLAB        │
        └─────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │ 3. Run MBAM (Functional) in MATLAB      │
        └─────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │ 4. Predict fMRI Eigenmodes (Python)     │
        └─────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │ 5. Reconstruct fMRI Maps (Python)       │
        └─────────────────────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │ 6. Diagnosis / Transfer Learning        │
        └─────────────────────────────────────────┘
```

# 🧪 Step-by-Step Pipeline

## 1. Generate Dummy Data

```bash
cd data
python step1_generate_mbm_dummy_data.py
python step2_generate_mbam_dummy_data.py
```

---

## 2. Structural MBM (MATLAB)

```matlab
cd matlab/mbm
step1_structural_mbm;
```

Output:

```
data/mbm_example/mbm_emp.mat
```

---

## 3. Functional MBAM (MATLAB)

```matlab
step2_functional_mbam;
```

Outputs:

```
data/mbam_example/MBAM/MBAM_data/MBAM_ckcpass_LR_30_164k_lh.mat
```

---

## 4. Predict fMRI Eigenmodes (Python)

```bash
python 01-struct2func_eigenmodes_prediction.py
```

Outputs:

```
results/functional_prediction/nn_struct2func/
```

---

## 5. Reconstruct fMRI Maps

```bash
python 02-eval_fmri_recon_from_pred_modes.py
```

Outputs:

```
results/functional_prediction/nn_struct2func/reconstruction/
```

---

## 6. Diagnosis Scripts (User Data Required)

Scripts:

```
03-classify_fa_vs_sca1_structural_modes.py
04-classify_fa_vs_sca3_structural_modes.py
05-classify_sca1_vs_sca3_structural_modes.py
```

---

## 7. Transfer Learning (User Data Required)

Scripts:

```
06-tl_classify_fa_vs_ctl_using_fmri_modes.py
07-tl_classify_fa_vs_ctl_using_structural_modes.py
08-tl_classify_fa_vs_ctl_multimodal.py
```

---

# 📁 Expected Data Directory

```
repo_root/
  data/
    mbm_example/
    mbam_example/
    diagnosis/
    transfer_learning/
```

---

# ✔️ Dummy vs User-Provided Data

| Component | Provided? | Used By |
|----------|-----------|---------|
| Dummy MBM/MBAM data | ✔️ Yes | MBM, MBAM, structure→function prediction |
| Diagnosis data | ❌ No | Scripts 03–05 |
| Transfer learning data | ❌ No | Scripts 06–08 |

---

# 🧠 Scientific Background & Citations

Geometric eigenmodes solve:

![laplace](https://latex.codecogs.com/svg.latex?\Delta\psi=-\lambda\psi)

Maps decompose as:

![recon](https://latex.codecogs.com/svg.latex?M=\sum_jc_j\psi_j)

Eigenmodes provide:

- biologically meaningful spatial basis  
- dimensionality reduction  
- structure–function linkage  

---

# 📜 How to Cite

```
@misc{ataxia-eigenmodes,
  title        = {ataxia-eigenmodes: Structural and functional brain geometry signatures for hereditary ataxias},
  author       = {Your Name and Collaborators},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/<your-username>/ataxia-eigenmodes}
}
```
