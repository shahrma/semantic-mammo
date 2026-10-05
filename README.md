# Semantic Feature Modulation for Mammographic Lesion Classification

Code accompanying the paper *Semantic Feature Modulation for Mammographic Lesion Classification*
(S. Mahpod and G. Ben Artzi, MICCAI 2026).

A ConvNeXt-Base classifier (benign vs. malignant) for mammography regions of interest (ROIs). BI-RADS semantic
descriptors — breast density, mass shape and margins, calcification morphology and distribution — condition the
backbone through **Conditional Layer Normalization**: selected LayerNorms in stage 3 of ConvNeXt are replaced by
`MetaLayerNorm` (`core/wrapped_models/wrapped_convnext.py`), whose scale and shift are predicted from the one-hot
descriptors (`Cond-LN Inter` in the paper). The baseline (`Base`) is the same network without conditioning.

This repository reproduces the CBIS-DDSM results table of the paper. It contains the code, the configurations
and **the train/test split used in the paper** (`folds/`).

## Repository layout

| Path | Content |
| --- | --- |
| `main.py` | training entry point |
| `core/` | dataset, augmentations, model, training loop and metrics |
| `configs/datasets/ddsm_{mass,calc,both}.yaml` | the three CBIS-DDSM subsets (all inherit `basesets.yaml`) |
| `configs/episodes/base.yaml`, `condln_inter_{mass,calc,both}.yaml` | Base and Cond-LN Inter models and the training schedule |
| `configs/categories/categories01.json` | fixed descriptor vocabularies that define the one-hot encodings (keep this file) |
| `folds/cbis_ddsm/` | the paper's train/test split, with `SHA256SUMS` |
| `scripts/utils/` | ROI extraction from the original CBIS-DDSM DICOM files |
| `checksums/cbis_ddsm_rois.sha256` | checksums of the ROI images used in the paper |
| `run_paper_experiments.sh` | trains the 30 runs of the table |

## 1. Installation

Tested on Linux with Python 3.12, PyTorch 2.6.0 and CUDA 12.4, on an NVIDIA GPU with 8 GB of memory.

```bash
git clone https://github.com/shahrma/semantic-mammo.git
cd semantic-mammo
conda create -n semammo python=3.12 -y
conda activate semammo
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.4 build of PyTorch. For another CUDA version, install the matching
`torch==2.6.0` / `torchvision==0.21.0` build first (see pytorch.org), then the rest of `requirements.txt`.
The ImageNet weights of ConvNeXt-Base are downloaded by torchvision on first use.

All commands below are run from the repository root.

## 2. Data

### 2.1 Download CBIS-DDSM

Download the CBIS-DDSM collection from The Cancer Imaging Archive (https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY):

* the DICOM images, with the NBIA Data Retriever using the "Descriptive Directory Name" layout (this also writes
  `metadata.csv`);
* the four case-description CSV files.

Arrange them as follows (`<CBIS_ROOT>` is any directory you choose):

```
<CBIS_ROOT>/
├── data/
│   ├── metadata.csv                       # written by the NBIA Data Retriever
│   └── CBIS-DDSM/
│       ├── Calc-Test_P_00038_LEFT_CC/...
│       └── ...
└── lists/
    ├── mass_case_description_train_set.csv
    ├── mass_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    └── calc_case_description_test_set.csv
```

### 2.2 Extract the ROIs

Each lesion is cropped with a fixed 1152 × 1152 px window centred on its mask and resized to 384 × 384 px
(16-bit PNG). `<WORKSET>` is the output directory:

```bash
python -c "
from scripts.utils.prepare_workset_cbis_ddsm import prepare_roi_workset
prepare_roi_workset('<CBIS_ROOT>', '<WORKSET>', borders=1152, border_type='center_fixed', output_size=384)"
```

This writes 3,568 ROIs to `<WORKSET>/rois` (plus `masks`, `full` and `meta`). It runs on the CPU (about 20 minutes on a
12-core laptop). Check that the ROIs are identical to those used in the paper:

```bash
(cd <WORKSET> && sha256sum -c --quiet "$OLDPWD/checksums/cbis_ddsm_rois.sha256") && echo "ROIs OK"
```

### 2.3 Add the paper's folds

The split is part of this repository and must be used as provided: it was drawn at random once and cannot be
regenerated. Copy it into the workset, where the training code reads it:

```bash
cp -r folds/cbis_ddsm <WORKSET>/folds
(cd <WORKSET>/folds && sha256sum -c --quiet SHA256SUMS) && echo "folds OK"
```

#### The split

```
folds/cbis_ddsm/
├── mass/            train_0.csv … train_4.csv, test.csv
├── calcification/   train_0.csv … train_4.csv, test.csv
└── SHA256SUMS
```

Each row is one ROI. `name` is the ROI file name in `<WORKSET>/rois`; `patient_id`, `pathology` (the label) and the
descriptor columns (`breast density`, `mass shape`, `mass margins`, `calc morphology`, `calc distribution`) come from
the CBIS-DDSM case descriptions. The training code reads labels and descriptors from these files.

Patients were assigned to folds at random, separately for each lesion type, so every patient of a subset is in
exactly one fold. The paper trains on the union of `train_0` … `train_4` and evaluates on `test`
(no cross-validation):

| Subset | Train (`train_0` … `train_4`) | Test (`test`) |
| --- | --- | --- |
| Mass | 1,401 ROIs, 740 patients (649 malignant) | 294 ROIs, 152 patients (134 malignant) |
| Calcification | 1,523 ROIs, 625 patients (566 malignant) | 349 ROIs, 128 patients (107 malignant) |
| Both (mass + calcification) | 2,924 ROIs | 643 ROIs |

The folds cover 3,567 of the 3,568 extracted ROIs (`P_00001_LEFT_CC_mass_1` is not used).

> **Note on the "Both" setting.** Because the mass and calcification splits were drawn independently, their
> union is not patient-disjoint: 22 patients have ROIs in both train and test (78 of the 643 test ROIs; for 19 of
> these, the same mammogram contributes a lesion of the other type to training). There is no patient overlap
> within the Mass or the Calcification subset. The folds are provided exactly as used, so that the published
> numbers can be reproduced.

## 3. Reproducing the results

### 3.1 Protocol

For each subset (Mass, Calc, Both) and model (Base, Cond-LN Inter), five runs are trained with seeds 42–46, for a
fixed 200 epochs. A run's score is the average over its last 20 epochs (181–200) of the test-set accuracy,
ROC-AUC, F1 and recall, with malignant as the positive class and the arg-max prediction. The table reports the
mean ± standard deviation over the five seeds. No checkpoint, epoch or threshold is selected.

### 3.2 Train

```bash
./run_paper_experiments.sh <WORKSET> ./workspace
```

This trains the 30 runs one after another. A subset can be run on its own, e.g.
`SUBSETS=mass SEEDS="42 43" ./run_paper_experiments.sh <WORKSET> ./workspace`.
A single run corresponds to:

```bash
python main.py --dataset ddsm_mass --episode condln_inter_mass --seed 42 \
    --name E01 --git_tag v1.4 --phase train --wandb_mode offline \
    --workspace ./workspace --params datasets.CBIS_DDSM.data_path:<WORKSET>
```

Use `--episode base` for the baseline, and `ddsm_calc`/`condln_inter_calc` or `ddsm_both`/`condln_inter_both` for
the other subsets. `--name E01 --git_tag v1.4` give the runs the same names as the paper runs. `<WORKSET>` must not
contain a `:` character.

Run time on one RTX 2070 (8 GB): about 7 h per Mass run (measured); Calc runs are of similar length and Both runs
take about twice as long. The original runs used Quadro RTX 6000 GPUs, five seeds at a time per GPU.

### 3.3 Examine the results

Each run writes `workspace/<group>/<run>/`, containing:

* `info/info_0001.pkl` … `info_0200.pkl`: for every epoch, the test-set class probabilities (`predictions`) and
  labels (`targets`);
* `last_model.pth`: the weights after epoch 200;
* `configuration.json`: the full configuration of the run;
* `wandb/`: the Weights & Biases log, with the per-epoch test accuracy (`val_acc`), ROC-AUC
  (`val_auc:pathology:malignant`), F1 (`val_f1`) and recall (`val_recall`).

The training script logs W&B offline by default. To browse the logs, run `wandb sync` on the run's `wandb/offline-run-*`
directory, or set `WANDB_MODE=online` (or pass `--wandb_mode online` to `main.py`) to log directly to your W&B
account. With `WANDB_MODE=disabled`, the metrics can be computed from the `info` files.

To obtain a table entry, average each run's metrics over epochs 181–200, then take the mean and standard deviation
over the five seeds (Section 3.1). `best_model.pth` is selected by test accuracy; it is not used for the reported
results and should not be used to report performance.

### 3.4 Expected results

CBIS-DDSM results reported in the paper (mean ± std in %, over 5 runs). `Cond-LN Inter` corresponds to the
`condln_inter_*` episodes.

| Subset | Model | Acc | AUC | F1 | Rec |
| --- | --- | --- | --- | --- | --- |
| Mass | Base | 80.3±1.1 | 88.7±0.8 | 78.5±1.4 | 79.1±2.2 |
| Mass | Cond-LN Inter | 84.7±0.9 | 90.9±1.3 | 83.0±0.9 | 81.8±2.7 |
| Calc | Base | 80.0±1.2 | 86.4±0.8 | 69.1±2.2 | 72.9±3.9 |
| Calc | Cond-LN Inter | 82.8±1.1 | 90.9±0.4 | 74.1±0.7 | 80.2±4.0 |
| Both | Base | 78.8±0.6 | 86.9±0.1 | 72.3±1.1 | 73.9±2.0 |
| Both | Cond-LN Inter | 81.2±0.7 | 90.3±0.4 | 75.0±1.7 | 75.4±4.3 |

Seeds and deterministic cuDNN are set, but training on a GPU still depends on the GPU model, driver, CUDA/cuDNN
and library versions. The original runs used Python 3.9 on Quadro RTX 6000 GPUs. On other hardware or software,
expect results to differ from the table by amounts comparable to the seed-to-seed standard deviation, rather than
bit-identical numbers.

## 4. Notes

* The descriptors are the radiologist annotations distributed with CBIS-DDSM. They are model inputs at both
  training and test time. Only the descriptors listed in the episode config are used; the pathology label and
  the other columns of the fold files (e.g. `assessment score`) are never model inputs.
* Descriptors that do not apply to a lesion type (e.g. mass shape for a calcification) and descriptors missing from
  the CBIS-DDSM annotations are both encoded as `N/A` (`core/datasets.py: refine_meta`). The vocabularies also
  contain an `unknown` entry, which no sample in these experiments uses.
* In the code, metrics logged as `val_*` are computed on the `test` fold (`folds_valid: test` in
  `configs/datasets/basesets.yaml`).
* `main.py` derives a run tag from `git describe` when `--git_tag` is not given, which requires a repository with
  at least one commit.

## Citation

```
@inproceedings{mahpod2026semantic,
  title     = {Semantic Feature Modulation for Mammographic Lesion Classification},
  author    = {Mahpod, Shahar and Ben Artzi, Gil},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```

If you use CBIS-DDSM, cite it as required by the TCIA Data Usage Policy:

* Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database
  for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive.
  <https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY>
* Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography
  data set for use in computer-aided detection and diagnosis research. Scientific Data, 4, 170177.
  <https://doi.org/10.1038/sdata.2017.177>
