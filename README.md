# WMH Segmentation with TrUE-Net

A complete pipeline for automated **White Matter Hyperintensity (WMH) segmentation** in brain MRI using **TrUE-Net** — a Triplanar Ensemble U-Net architecture. The pipeline covers preprocessing, inference (FLAIR-only, T1-only, or dual-channel), and post-processing volume quantification.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Triplanar U-Net](#triplanar-u-net)
   - [Spatial Weight Maps](#spatial-weight-maps)
   - [Fine-Tuning Layers](#fine-tuning-layers)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Pipeline: Step-by-Step](#pipeline-step-by-step)
   - [Step 1 — Create Path CSVs](#step-1--create-path-csvs)
   - [Step 2a — FLAIR-Based Inference](#step-2a--flair-based-inference)
   - [Step 2b — T1-Based Inference](#step-2b--t1-based-inference)
   - [Step 3 — Calculate WMH Volumes](#step-3--calculate-wmh-volumes)
8. [TrUE-Net CLI Reference](#true-net-cli-reference)
   - [train](#train)
   - [evaluate](#evaluate)
   - [fine_tune](#fine_tune)
   - [cross_validate](#cross_validate)
9. [Fine-Tuning Guide](#fine-tuning-guide)
10. [Output Description](#output-description)
11. [Configuration Reference](#configuration-reference)
12. [Tips & Troubleshooting](#tips--troubleshooting)
13. [Citation](#citation)
14. [License](#license)

---

## Overview

White Matter Hyperintensities (WMH) are bright lesion regions visible on FLAIR MRI scans, associated with cerebrovascular disease and cognitive decline. This project provides:

- **Automated WMH segmentation** using pretrained TrUE-Net models (UKBB FLAIR and UKBB T1 weights)
- **Multi-modal inference** supporting FLAIR-only, T1-only, or combined FLAIR+T1 inputs
- **SynthSeg-assisted registration** to bring T1-derived anatomical priors into FLAIR space
- **Volume quantification** from output probability maps, producing per-subject, per-timepoint CSV output

This pipeline was applied to longitudinal OASIS-3 MRI data (`sub-OAS3XXXX_sess-dXXXX` naming convention).

---

## Architecture

### Triplanar U-Net

TrUE-Net applies three independent 2D U-Nets — one for each anatomical plane (axial, sagittal, coronal) — and averages their 3D probability volumes to produce the final segmentation map.

![TrUE-Net Main Architecture](assets/main_architecture_final.png)

Each 2D U-Net receives a 2-channel input (FLAIR + T1) and outputs per-voxel WMH probability scores:

| Plane    | Input Size  | Channels        |
|----------|-------------|-----------------|
| Axial    | 128 × 192   | FLAIR + T1      |
| Sagittal | 192 × 120   | FLAIR + T1      |
| Coronal  | 128 × 80    | FLAIR + T1      |

The three resulting 3D volumes are assembled from 2D slices, resized back to the original dimensions, and averaged to produce the **Final Probability Map**.

Convolution types used in the architecture:

- **Orange arrow** — 1×1 convolution
- **Dark blue arrow** — 3×3 conv + Batch Norm + ReLU
- **White arrow** — 3×3 conv (axial) / 5×5 conv (sagittal & coronal) + Batch Norm + ReLU
- **Red arrow** — Max-pooling 2×2
- **Green arrow** — Up-conv 2×2 (output channels = input / 2)
- **Grey arrow** — Copy and concatenate (skip connection)

---

### Spatial Weight Maps

To bias the network toward anatomically plausible WMH regions, TrUE-Net uses a **spatially weighted loss function**. The spatial weight map is computed as the sum of two distance maps derived from the T1 segmentation:

![Spatial Weight Map](assets/spatial_weight_map.png)

**Spatial Distance Map = Ventricle Distance Map + GM Distance Map**

- The **Ventricle Distance Map** highlights periventricular regions (high WMH prevalence).
- The **GM Distance Map** down-weights cortical grey matter (unlikely to contain WMH).

These maps must be pre-computed and provided via the `-gdir` and `-vdir` flags during training or fine-tuning when using the `weighted` loss function.

---

### Fine-Tuning Layers

The U-Net encoder-decoder is divided into 8 numbered layers for selective fine-tuning. Layers are counted from the output side (decoder) inward:

![Fine-Tuning Layer Numbers](assets/fine_tuning_images.png)

| Layer | Location          | Feature Channels |
|-------|-------------------|------------------|
| 1     | Output (1×1 conv) | 2                |
| 2     | Decoder top       | 128, 64, 64      |
| 3     | Decoder mid-upper | 256, 128         |
| 4     | Decoder mid-lower | 512, 256         |
| 5     | Bottleneck        | 512              |
| 6     | Encoder mid-lower | 256, 256         |
| 7     | Encoder mid-upper | 128, 128         |
| 8     | Encoder top       | 2, 3, 64, 64     |

To fine-tune only the last two decoder layers, use `-ftlayers 1 2`. To fine-tune deeper, add more layer numbers (e.g., `-ftlayers 1 2 3 4`).

---

## Repository Structure

```
wmh-segmentation-truenet/
├── README.md
├── assets/                         # Architecture diagrams
│   ├── main_architecture_final.png
│   ├── fine_tuning_images.png
│   └── spatial_weight_map.png
│
├── src/
│   ├── truenet.py                  # TrUE-Net CLI entry point
│   ├── create_paths_csvs.py        # Build image path CSVs for batch inference
│   └── calculate_wmh_volumes.py    # Compute WMH volumes from probability maps
│
├── scripts/
│   ├── Truenet_Flair.sh            # Batch FLAIR inference (SynthSeg-registered)
│   ├── truenet_flairwmhinfer.sh    # Simple single-run FLAIR inference
│   └── truenet_T1.sh               # Single T1 image inference
│
├── models/
│   ├── ukbb_flair/                 # Pretrained FLAIR model weights
│   └── ukbb_t1/                    # Pretrained T1 model weights
│
├── CSV_Lists/                      # Auto-generated by create_paths_csvs.py
│   ├── flair_synthseg_images.csv
│   └── t1_images.csv
│
└── outputs/
    ├── WMH_Flair_SynthSeg/         # Per-subject FLAIR segmentation maps
    ├── WHM_Seg_Maps_T1w/           # Per-subject T1 segmentation maps
    └── wmh_volumes_T1w.csv         # Final aggregated volume table
```

---

## Prerequisites

| Requirement        | Details                                               |
|--------------------|-------------------------------------------------------|
| OS                 | Linux (Ubuntu 18.04+ recommended)                     |
| Python             | 3.7+ (FSL bundled Python recommended)                 |
| FSL                | 6.0+ (provides `truenet`, bias correction tools)      |
| GPU                | CUDA-capable GPU strongly recommended for inference   |
| Python packages    | `nibabel`, `numpy`, `pandas`, `natsort`, `torch`      |

> **Note:** This pipeline was developed using the FSL-bundled Python at `/home/biomedialab/fsl/bin/python`. If you use a different Python environment, update the `PYTHON_EXEC` variable in all shell scripts accordingly.

---

## Installation

**1. Install FSL (includes TrUE-Net):**

```bash
# Follow official FSL installation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
```

**2. Install the standalone `truenet` package (if not using FSL):**

```bash
pip install truenet
```

**3. Install Python dependencies:**

```bash
pip install nibabel numpy pandas natsort
```

**4. Clone this repository:**

```bash
git clone https://github.com/your-username/wmh-segmentation-truenet.git
cd wmh-segmentation-truenet
```

**5. Download pretrained model weights:**

Place the pretrained weights into the `models/` directory:

```
models/
├── ukbb_flair/    # UKBB FLAIR-trained weights
└── ukbb_t1/       # UKBB T1-trained weights
```

Weights are available from the official TrUE-Net release:
[https://github.com/v-sundaresan/truenet](https://github.com/v-sundaresan/truenet)

---

## Data Preparation

### File Naming Convention

TrUE-Net requires strict file suffixes to identify modalities inside an input directory:

| Modality | Required suffix         | Example                              |
|----------|-------------------------|--------------------------------------|
| FLAIR    | `_FLAIR.nii.gz`         | `sub-OAS30001_sess-d0129_FLAIR.nii.gz` |
| T1       | `_T1.nii.gz`            | `sub-OAS30001_sess-d0129_T1.nii.gz`    |

> **Important:** Both files for a subject must reside in the **same input directory**. The `truenet_T1.sh` script handles this automatically by creating a temporary symlink with the required `_T1.nii.gz` suffix.

### Preprocessing Checklist

Before running inference, ensure your images have been:

- [x] **Bias field corrected** — e.g., using `fsl_anat` or `N4BiasFieldCorrection`
- [x] **Skull-stripped** — brain extraction is required for reliable results
- [x] **Co-registered** — T1 must be registered into FLAIR space (or vice versa)
  - This pipeline uses **SynthSeg-based linear registration** to produce `*_T1w_synthseg_in_FLAIR.nii.gz` images
- [x] **Spatial weight maps generated** — required for training/fine-tuning with weighted loss

### OASIS-3 Specific Notes

This pipeline uses the OASIS-3 naming convention: `sub-OAS3XXXX_sess-dXXXX`. The `calculate_wmh_volumes.py` script parses subject IDs and visit times from this pattern automatically. If your data uses a different naming scheme, update the `extract_info()` function in `calculate_wmh_volumes.py`.

---

## Pipeline: Step-by-Step

### Step 1 — Create Path CSVs

`create_paths_csvs.py` scans a directory for MRI files and writes their paths to a CSV, which the shell scripts and `truenet apply` command consume.

**Configure paths at the top of the script:**

```python
# Input: directory containing SynthSeg-registered FLAIR images
SEARCH_DIR = "/path/to/FLAIR_SPACE_OUTPUT"

# Output: where to save the CSV file
OUTPUT_CSV_DIR = "/path/to/CSV_Lists"
```

**Run:**

```bash
python src/create_paths_csvs.py
```

**Output:** `CSV_Lists/flair_synthseg_images.csv` — one file path per row under the header `FLAIR`.

> To generate a T1 image list, uncomment the original `create_csvs()` block at the top of the file. It will produce both `flair_images.csv` and `t1_images.csv`.

---

### Step 2a — FLAIR-Based Inference

Two scripts are provided for FLAIR inference, depending on your workflow:

#### Option A: Batch Inference (Recommended)

`scripts/Truenet_Flair.sh` reads paths from the CSV generated in Step 1 and runs TrUE-Net on each subject.

**Configure:**

```bash
csv_file="/path/to/CSV_Lists/flair_synthseg_images.csv"
base_output_dir="/path/to/WMH_Flair_SynthSeg/"
model_path="/path/to/models/ukbb_flair/"
PYTHON_EXEC="/home/biomedialab/fsl/bin/python"
TRUENET_SCRIPT="/home/biomedialab/fsl/bin/truenet"
```

**Run:**

```bash
bash scripts/Truenet_Flair.sh
```

Per-subject output folders are created automatically: `WMH_Flair_SynthSeg/sub-OAS3XXXX/`.

#### Option B: Simple Single-Run Inference

`scripts/truenet_flairwmhinfer.sh` uses the `truenet apply` command directly with the pretrained `ukbb_flair` model:

```bash
bash scripts/truenet_flairwmhinfer.sh
```

> This script uses the `truenet apply` shorthand and references `ukbb_flair` as a named built-in model.

---

### Step 2b — T1-Based Inference

`scripts/truenet_T1.sh` runs inference on a **single** T1 image. It creates a temporary symlink with the `_T1.nii.gz` suffix required by TrUE-Net, runs evaluation, then cleans up.

**Configure:**

```bash
T1_IMAGE="/path/to/your/bet_corrected_mri.nii.gz"
OUTPUT_DIR="/path/to/Single_T1_Output"
MODEL_PATH="/path/to/models/ukbb_t1/"
PYTHON_EXEC="/home/biomedialab/fsl/bin/python"
TRUENET_SCRIPT="/home/biomedialab/fsl/bin/truenet"
```

**Run:**

```bash
bash scripts/truenet_T1.sh
```

> To run T1 inference across multiple subjects, adapt `Truenet_Flair.sh` by pointing it at a T1 CSV and the `ukbb_t1` model path.

---

### Step 3 — Calculate WMH Volumes

`src/calculate_wmh_volumes.py` reads all `*.nii.gz` probability maps from a segmentation output directory, thresholds them at 0.45 (voxels with probability ≥ 0.45 are counted as WMH), multiplies by voxel volume, and writes a summary CSV.

**Configure:**

```python
# Directory containing your segmentation maps
INPUT_DIR = "/path/to/WHM_Seg_Maps_T1w/"

# Where to save the results
OUTPUT_CSV = "/path/to/wmh_volumes_T1w.csv"
```

**Run:**

```bash
python src/calculate_wmh_volumes.py
```

**Output CSV columns:**

| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| `Subject_ID`       | Parsed from filename (e.g., `OAS30001`)          |
| `Visit_Time_Days`  | Integer days since baseline (from `sess-dXXXX`)  |
| `Volume_mm3`       | WMH volume in cubic millimetres                  |
| `Volume_mL`        | WMH volume in millilitres (mm3 / 1000)           |

Results are sorted by `Subject_ID` then `Visit_Time_Days` for longitudinal analysis.

> **Threshold note:** The default threshold is 0.45 (i.e., `data >= 0.45`). Adjust this in `calculate_wmh_volumes.py` if your use case requires a different sensitivity/specificity trade-off.

---

## TrUE-Net CLI Reference

All four TrUE-Net commands are accessed via `truenet.py` (or the `truenet` FSL command):

```
truenet {train, evaluate, fine_tune, cross_validate} [options]
```

---

### `train`

Train TrUE-Net from scratch on your dataset.

```bash
truenet train \
  -i  <input_dir>        # Directory with _FLAIR.nii.gz and _T1.nii.gz images
  -m  <model_dir>        # Directory to save model weights
  -l  <label_dir>        # Directory with manual lesion masks (default: inp_dir)
```

Key optional arguments:

| Flag              | Default   | Description                                              |
|-------------------|-----------|----------------------------------------------------------|
| `-tr_prop`        | 0.8       | Fraction of data used for training (rest = validation)   |
| `-loss`           | weighted  | Loss type: `weighted` (spatially weighted) or `nweighted`|
| `-gdir`           | inp_dir   | GM distance map directory (required if `weighted`)       |
| `-vdir`           | inp_dir   | Ventricle distance map directory (required if `weighted`)|
| `-plane`          | all       | Plane(s) to train: `axial`, `sagittal`, `coronal`, `all`|
| `-da`             | True      | Apply data augmentation (pass flag to disable)           |
| `-af`             | 2         | Augmentation inflation factor                            |
| `-bs`             | 8         | Batch size                                               |
| `-ep`             | 60        | Number of epochs                                         |
| `-es`             | 20        | Early stopping patience (epochs)                         |
| `-ilr`            | 0.001     | Initial learning rate                                    |
| `-opt`            | adam      | Optimizer: `adam` or `sgd`                               |
| `-cp_type`        | last      | Checkpoint saving: `best`, `last`, or `everyN`           |
| `-fo`             | —         | Use FLAIR only (flag, no value)                          |
| `-to`             | —         | Use T1 only (flag, no value)                             |
| `-cpu`            | —         | Force CPU (no GPU)                                       |
| `-v`              | —         | Verbose/debug output                                     |

**Example:**

```bash
truenet train \
  -i /data/training_subjects/ \
  -m /models/my_model/ \
  -l /data/manual_masks/ \
  -loss weighted \
  -gdir /data/gm_dist_maps/ \
  -vdir /data/vent_dist_maps/ \
  -ep 80 -bs 8 -ilr 0.0005 \
  -cp_type best -v
```

---

### `evaluate`

Run inference with a saved or pretrained model.

```bash
truenet evaluate \
  -i  <input_dir>    # Directory with test images
  -m  <model_name>   # Pretrained model name or path to model weights
  -o  <output_dir>   # Directory for saving probability maps
```

Key optional arguments:

| Flag         | Default | Description                                              |
|--------------|---------|----------------------------------------------------------|
| `-int`       | False   | Save per-plane intermediate predictions                  |
| `-cp_type`   | last    | Which checkpoint to load: `best`, `last`, `specific`     |
| `-cp_n`      | —       | Epoch N to load (only if `-cp_type specific`)            |
| `-fo`        | —       | FLAIR-only inference                                     |
| `-to`        | —       | T1-only inference                                        |
| `-cpu`       | —       | Force CPU                                                |
| `-v`         | —       | Verbose output                                           |

**Example (with SynthSeg FLAIR input):**

```bash
truenet evaluate \
  -i /data/subject_flair_dir/ \
  -m /models/ukbb_flair/ \
  -o /outputs/wmh_maps/ \
  -fo -v
```

---

### `fine_tune`

Fine-tune a pretrained model on your own data using selective layer unfreezing.

```bash
truenet fine_tune \
  -i  <input_dir>    # Directory with training images
  -m  <model_name>   # Pretrained model to start from
  -o  <output_dir>   # Directory to save fine-tuned weights
  -l  <label_dir>    # Directory with manual lesion masks
```

Key optional arguments:

| Flag          | Default   | Description                                            |
|---------------|-----------|--------------------------------------------------------|
| `-ftlayers`   | 2         | Layer numbers to fine-tune (see [Fine-Tuning Layers](#fine-tuning-layers)) |
| `-ilr`        | 0.0001    | Initial learning rate (lower than training, recommended) |
| `-loss`       | weighted  | Loss function type                                     |
| `-gdir`       | inp_dir   | GM distance map directory                              |
| `-vdir`       | inp_dir   | Ventricle distance map directory                       |
| `-plane`      | all       | Which plane(s) to fine-tune                            |
| `-ep`         | 60        | Number of fine-tuning epochs                           |
| `-es`         | 20        | Early stopping patience                                |
| `-cpld_type`  | last      | Which pretrained checkpoint to load                    |

**Example — fine-tune only the last 3 decoder layers:**

```bash
truenet fine_tune \
  -i /data/local_subjects/ \
  -m /models/ukbb_flair/ \
  -o /models/finetuned_local/ \
  -l /data/local_masks/ \
  -ftlayers 1 2 3 \
  -ilr 0.0001 -ep 40 -es 15 -v
```

---

### `cross_validate`

Perform k-fold cross-validation for model evaluation.

```bash
truenet cross_validate \
  -i  <input_dir>    # Directory with all subject images
  -o  <output_dir>   # Directory for saving predictions and (optionally) models
  -l  <label_dir>    # Directory with manual lesion masks
```

Key optional arguments:

| Flag           | Default | Description                                          |
|----------------|---------|------------------------------------------------------|
| `-fold`        | 5       | Number of cross-validation folds                     |
| `-resume_fold` | 1       | Resume from a specific fold (if interrupted)         |
| `-sv`          | False   | Save model checkpoints during cross-validation       |
| `-int`         | False   | Save per-plane intermediate predictions              |
| `-loss`        | weighted| Loss function type                                   |

**Example:**

```bash
truenet cross_validate \
  -i /data/all_subjects/ \
  -o /outputs/cv_results/ \
  -l /data/manual_masks/ \
  -fold 5 -ep 60 -v
```

---

## Fine-Tuning Guide

Fine-tuning is recommended when your data differs from the UKBB cohort (e.g., different scanner, different age group, different pathology severity).

**General strategy:**

1. **Start shallow** — fine-tune only layers 1–2 (output + top decoder) on small datasets.
2. **Go deeper** if you have more annotated subjects — add layers 3–4 for mid-level features.
3. **Use a very low learning rate** — default is `0.0001` (10× lower than training).
4. **Keep weighted loss** if your dataset has spatial bias maps available; switch to `nweighted` otherwise.

**Minimum recommended dataset sizes for fine-tuning:**

| Fine-tuned layers | Minimum subjects |
|-------------------|------------------|
| 1–2 (shallow)     | ~10–20           |
| 1–4 (mid-depth)   | ~30–50           |
| All 8 layers      | ~100+            |

Refer to the [layer diagram](#fine-tuning-layers) for a visual guide to layer depth.

---

## Output Description

TrUE-Net inference produces the following output for each subject:

```
subject_output_dir/
└── Predicted_probmap_truenet_<subject_name>.nii.gz
```

This is a **3D probability map** in NIfTI format with the same dimensions and affine as the input images. Each voxel value represents the model's confidence (0–1) that it contains WMH tissue.

To convert to a binary mask, threshold at 0.5:

```python
import nibabel as nib
import numpy as np

img = nib.load("Predicted_probmap_truenet_sub-OAS30001_sess-d4467.nii.gz")
binary_mask = (img.get_fdata() >= 0.5).astype(np.uint8)
nib.save(nib.Nifti1Image(binary_mask, img.affine), "wmh_mask.nii.gz")
```

`calculate_wmh_volumes.py` uses a threshold of **0.45** by default to be slightly more inclusive. Adjust as needed.

---

## Configuration Reference

A summary of all hard-coded paths you need to update before running:

| Script / File                  | Variable             | Description                            |
|-------------------------------|----------------------|----------------------------------------|
| `create_paths_csvs.py`        | `SEARCH_DIR`         | Root folder containing MRI images      |
| `create_paths_csvs.py`        | `OUTPUT_CSV_DIR`     | Where to write CSV files               |
| `Truenet_Flair.sh`            | `csv_file`           | Path to FLAIR CSV list                 |
| `Truenet_Flair.sh`            | `base_output_dir`    | Root for per-subject output folders    |
| `Truenet_Flair.sh`            | `model_path`         | Path to pretrained FLAIR model         |
| `Truenet_Flair.sh`            | `PYTHON_EXEC`        | Path to Python executable              |
| `Truenet_Flair.sh`            | `TRUENET_SCRIPT`     | Path to truenet script                 |
| `truenet_T1.sh`               | `T1_IMAGE`           | Path to single T1 input image          |
| `truenet_T1.sh`               | `OUTPUT_DIR`         | Output directory for single T1 run     |
| `truenet_T1.sh`               | `MODEL_PATH`         | Path to pretrained T1 model            |
| `calculate_wmh_volumes.py`    | `INPUT_DIR`          | Directory containing probability maps  |
| `calculate_wmh_volumes.py`    | `OUTPUT_CSV`         | Path for the final volumes CSV         |

---

## Tips & Troubleshooting

**TrUE-Net cannot find input images**
> Ensure filenames end exactly with `_FLAIR.nii.gz` and/or `_T1.nii.gz`. Both files must be in the same input directory (not subdirectories).

**`truenet_T1.sh` produces an empty output**
> The script creates a temporary symlink. Ensure the `T1_IMAGE` path is absolute (not relative) so the `ln -sf` command works correctly from any working directory.

**Volume script finds 0 files**
> Check that the `INPUT_DIR` path is correct and that files are named `*.nii.gz`. The script uses a recursive glob, so nested directories are supported.

**Unexpected high/low WMH volumes**
> Adjust the probability threshold in `calculate_wmh_volumes.py` (currently `>= 0.45`). Higher thresholds (e.g., 0.5) yield more conservative (smaller) volumes; lower thresholds (e.g., 0.35) yield more inclusive volumes.

**CUDA out of memory during inference**
> Reduce the batch size with `-bs 4` or `-bs 2`, or add `-cpu` to run on CPU (slower but memory-safe).

**Subject ID parsing fails in volume script**
> The `extract_info()` function expects filenames containing `sub-<ID>` and `sess-d<days>`. If your data uses a different convention, update the regex patterns in that function.

---

## Citation

If you use this pipeline or TrUE-Net in your research, please cite the original paper:

```bibtex
@article{sundaresan2021automated,
  title={Automated lesion segmentation with FLAIR MRI: combined unsupervised and supervised methods},
  author={Sundaresan, Vaanathi and Zamboni, Giovanna and Rothwell, Peter M and Jenkinson, Mark and Griffanti, Ludovica},
  journal={IEEE Transactions on Medical Imaging},
  year={2021}
}
```

For the full TrUE-Net preprint:
> Sundaresan et al. (2020). *Triplanar Ensemble U-Net model for white matter hyperintensities segmentation on MR images.*
> bioRxiv. https://www.biorxiv.org/content/10.1101/2020.07.24.219485v1

---

## License

Specify your preferred license here (e.g., MIT, Apache 2.0, or GNU GPL v3).

TrUE-Net itself is distributed under the Apache 2.0 License. See [https://github.com/v-sundaresan/truenet](https://github.com/v-sundaresan/truenet) for details.