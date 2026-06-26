# Rogii Wellbore Geology Prediction (Kaggle)

> A LightGBM-based regression pipeline to predict True Vertical Thickness (TVT) for horizontal wellbore geology, built for the Rogii Wellbore Geology Prediction Kaggle competition.

---

## 📌 Overview

This project tackles a **subsurface geology prediction task**: given well-log measurements along a horizontal wellbore, predict the **True Vertical Thickness (TVT)** at each measured depth. The pipeline handles per-well data cleaning, missing-value imputation, and trains a **10-fold LightGBM regressor** to generate competition submissions.

---

## 🏗️ Pipeline

```
Raw well CSVs (train/test, per-well files)
        │
        ▼
  Load & concatenate horizontal well + type well data
        │
        ▼
  Missing value handling:
    - Replace inf/-inf with NaN
    - Forward-fill + backward-fill per well_id
    - Global median fill for remaining gaps
        │
        ▼
  10-Fold KFold LightGBM Regression
        │
        ▼
  Average fold predictions → map to sample_submission.csv
```

---

## ⚙️ Key Details

| Component | Choice | Notes |
|---|---|---|
| **Features** | `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input` | Measured depth, spatial coordinates, gamma ray, and input TVT |
| **Target** | `TVT` | True Vertical Thickness |
| **Model** | LightGBM Regressor | 2500 estimators, lr=0.01, max_depth=10, num_leaves=127 |
| **Validation** | 10-Fold KFold (shuffled) | Predictions averaged across all folds |
| **Early Stopping** | 50 rounds | Based on validation set performance per fold |
| **Imputation Strategy** | Per-well ffill/bfill → global median fallback | Preserves well-specific trends before falling back to a global statistic |

---

## 📂 Data

Data is provided by the Kaggle competition as per-well CSV files, split into:
- **Horizontal well files** — the wells being predicted on
- **Type well files** — reference geology used for context

Each well file is identified by a `well_id` parsed from the filename, and horizontal/type well files are concatenated separately before merging on common columns.

> Data is **not included** in this repository — see [How to Run](#-how-to-run) for download instructions.

---

## 🚀 How to Run

This notebook was built for the **Kaggle** environment (`/kaggle/input/`, `/kaggle/working/` paths).

### 1. Clone the repository
```bash
git clone https://github.com/daffbg/rogii-wellbore-geology-prediction.git
cd rogii-wellbore-geology-prediction
```

### 2. Get the competition data
Download the dataset from the [Rogii Wellbore Geology Prediction competition page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) on Kaggle, or run this notebook directly on Kaggle where the data is already mounted at `/kaggle/input/competitions/rogii-wellbore-geology-prediction/`.

### 3. Install dependencies
```bash
pip install pandas numpy lightgbm scikit-learn tqdm
```

### 4. Run the notebook
```bash
jupyter notebook rogii_wellbore_geology_prediction.ipynb
```

> If running outside Kaggle, update the `base_path` variable to point to your local copy of the dataset.

---

## 🛠️ Requirements

```
pandas
numpy
lightgbm
scikit-learn
tqdm
```

---

## 📤 Output

The notebook generates `submission.csv` containing predicted TVT values mapped to the competition's `sample_submission.csv` format (`id`, `tvt`).

---

## 👤 Author

**Sudipta Mondal**
GitHub: [github.com/daffbg](https://github.com/daffbg)
LinkedIn: [linkedin.com/in/sudipta-mondal-4553512a2](https://linkedin.com/in/sudipta-mondal-4553512a2)
