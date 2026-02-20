# Classification — CNN on CIFAR-10 & Ensemble Methods on Adult Income

A two-part classification project: **(1)** a custom CNN for 10-class image classification on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (60,000 images), and **(2)** Random Forest, SVM, and XGBoost for binary income prediction on the [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) dataset (48,842 records).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1: CNN Image Classification (CIFAR-10)](#part-1-cnn-image-classification-cifar-10)
  - [Dataset](#cifar-10-dataset)
  - [CNN Architecture](#cnn-architecture)
  - [Training Configuration](#training-configuration)
  - [CNN Optimisation Techniques](#cnn-optimisation-techniques)
- [Part 2: Tabular Classification (Adult Income)](#part-2-tabular-classification-adult-income)
  - [Dataset](#adult-income-dataset)
  - [Feature Descriptions](#feature-descriptions)
  - [Data Preprocessing](#data-preprocessing)
  - [Models](#models)
  - [Tabular Optimisation Techniques](#tabular-optimisation-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Prediction & Inference](#prediction--inference)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Project Overview

| Part | Task | Dataset | Models | Classes |
|------|------|---------|--------|---------|
| **1** | Image Classification | CIFAR-10 (60K images, 32×32 RGB) | Custom CNN | 10 (airplane, automobile, bird, …) |
| **2** | Tabular Classification | Adult Income (48,842 rows, 14 features) | Random Forest, SVM, XGBoost | 2 (≤50K, >50K) |

---

## Part 1: CNN Image Classification (CIFAR-10)

### CIFAR-10 Dataset

| Property | Value |
|----------|-------|
| Source | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, Nair & Hinton) |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32 × 32 × 3 (RGB) |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

Loaded directly from `keras.datasets.cifar10` — no manual download required.

### CNN Architecture

A three-block convolutional network with batch normalisation and progressive dropout:

```
Input (32×32×3)
    │
Data Augmentation (horizontal flip, rotation, zoom, translation)
    │
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    │
Flatten → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
    │
Dense(10, Softmax) → Output
```

| Component | Details |
|-----------|---------|
| Convolutional blocks | 3 blocks, each with 2× Conv2D + BatchNorm + ReLU |
| Filters | 32 → 64 → 128 (progressive widening) |
| Kernel size | 3 × 3 with `same` padding |
| Pooling | MaxPooling 2 × 2 after each block |
| Dense head | 512 units with BatchNorm and 50% dropout |
| Output | 10-class softmax |
| Loss | Categorical cross-entropy |
| Optimiser | Adam (LR = 1e-3) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Epochs | 50 (with early stopping) |
| Validation split | 20% of training data |
| Early stopping | Patience 10, restore best weights |
| LR scheduling | `ReduceLROnPlateau` — factor 0.5, patience 5, min LR 1e-6 |

### CNN Optimisation Techniques

| # | Technique | Details |
|---|-----------|---------|
| 1 | **Data Augmentation** | Horizontal flip, ±10% rotation, ±10% zoom, ±10% translation |
| 2 | **Batch Normalisation** | After every convolutional and first dense layer |
| 3 | **Dropout** | 25% after each conv block, 50% before output |
| 4 | **Early Stopping** | Patience 10 on `val_loss`, restores best weights |
| 5 | **Learning Rate Scheduling** | Halves LR when validation loss plateaus |

---

## Part 2: Tabular Classification (Adult Income)

### Adult Income Dataset

| Property | Value |
|----------|-------|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) (1994 Census) |
| Samples | 48,842 |
| Features | 14 (6 continuous + 8 categorical) |
| Target | Income — binary (>50K or ≤50K) |
| Class balance | Imbalanced (~75% ≤50K, ~25% >50K) |

Loaded directly from the UCI archive URL — no manual download required.

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Continuous | Age of the individual |
| `workclass` | Categorical | Type of employer (Private, Gov, Self-employed, etc.) |
| `fnlwgt` | Continuous | Census sampling weight |
| `education` | Categorical | Highest education level achieved |
| `education_num` | Continuous | Numeric representation of education level |
| `marital_status` | Categorical | Marital status |
| `occupation` | Categorical | Type of occupation |
| `relationship` | Categorical | Relationship status in household |
| `race` | Categorical | Race of the individual |
| `sex` | Categorical | Gender |
| `capital_gain` | Continuous | Capital gains recorded |
| `capital_loss` | Continuous | Capital losses recorded |
| `hours_per_week` | Continuous | Hours worked per week |
| `native_country` | Categorical | Country of origin |

### Data Preprocessing

1. **Missing value handling** — rows with `?` values dropped
2. **Target encoding** — `>50K` → 1, `<=50K` → 0
3. **Label encoding** — all categorical features encoded to integers via `LabelEncoder`
4. **Stratified split** — 80/20 train/test with `random_state=42`, preserving class proportions
5. **Feature scaling** — `StandardScaler` (zero mean, unit variance)

### Models

All three models are trained with manually tuned hyperparameters:

#### Random Forest

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 15 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | sqrt |

#### Support Vector Machine (SVM)

| Parameter | Value |
|-----------|-------|
| `kernel` | RBF |
| `C` | 1.0 |
| `gamma` | scale |
| `probability` | True |

#### XGBoost

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |

### Tabular Optimisation Techniques

| # | Technique | Details |
|---|-----------|---------|
| 1 | **Feature Scaling** | `StandardScaler` — essential for SVM |
| 2 | **RF Regularisation** | `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2` |
| 3 | **SVM Kernel** | RBF with automatic gamma — handles non-linear boundaries |
| 4 | **XGBoost Regularisation** | L1/L2 regularisation + subsampling (rows and columns) |
| 5 | **Stratified Splitting** | Preserves class proportions in train/test sets |

---

## Evaluation Metrics

Both parts are evaluated with the same five metrics:

| Metric | Averaging | Description |
|--------|-----------|-------------|
| **Accuracy** | — | Overall fraction of correct predictions |
| **Precision** | Weighted (Part 1) / Binary (Part 2) | Proportion of positive predictions that are correct |
| **Recall** | Weighted / Binary | Proportion of actual positives correctly identified |
| **F1-Score** | Weighted / Binary | Harmonic mean of precision and recall |
| **AUC-ROC** | Weighted OvR (Part 1) / Binary (Part 2) | Area under the ROC curve |

### Visualisations

The notebook produces:

- **CIFAR-10 sample images** — one image per class
- **CNN training curves** — accuracy and loss over epochs
- **CNN confusion matrix** — 10 × 10 heatmap with class labels
- **CNN classification report** — per-class precision, recall, F1
- **Income class distribution** — bar chart of ≤50K vs >50K
- **Tabular model comparison** — grouped bar chart of all metrics across RF, SVM, XGBoost
- **Tabular confusion matrices** — side-by-side heatmaps for all three models
- **XGBoost feature importance** — ranked bar chart of all 14 features
- **CNN prediction samples** — test images with predicted vs true labels, correct/incorrect markers

---

## Prediction & Inference

### Image Classification

```python
predicted_class, confidence = predict_image_class(
    image=test_image,        # 32×32×3 normalised array
    model=cnn_model,
    class_names=class_names,
)
# Output: "automobile", 94.2%
```

### Income Prediction

The tabular models predict directly on new scaled feature vectors:

```python
prediction = xgb_model.predict(scaler.transform(new_data))
# Output: 1 (>50K) or 0 (<=50K)
```

---

## Getting Started

### Requirements

- **Hardware:** GPU recommended for CNN training (Google Colab with T4/A100 or local CUDA GPU)
- **Python:** 3.8+

### Installation

```bash
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn
```

### Running

1. Open `Classification_CNN__Random_Forest__SVM__XGBoost__on_Adult_Income.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially.
3. Both datasets are loaded automatically — CIFAR-10 via Keras, Adult Income via UCI URL.

---

## Notebook Structure

| Cell(s) | Section | Description |
|---------|---------|-------------|
| 0–2 | Setup | Introduction, install dependencies, import libraries |
| **3–14** | **Part 1: CNN (CIFAR-10)** | |
| 3–5 | Data loading | Load CIFAR-10, dataset info, visualise sample images |
| 6–7 | Preprocessing | Normalise pixels, one-hot encode labels, data augmentation layer |
| 8–9 | Model building | 3-block CNN architecture, compile with Adam, define callbacks |
| 10 | Training | Train for 50 epochs with early stopping and LR scheduling |
| 11 | Training curves | Accuracy and loss plots |
| 12–14 | Evaluation | Test metrics, confusion matrix, per-class classification report |
| **15–33** | **Part 2: Tabular (Adult Income)** | |
| 15–20 | Data loading & EDA | Load from UCI, feature descriptions, statistics, missing values |
| 21–22 | Preprocessing | Drop missing, encode target, visualise class distribution |
| 23–25 | Feature engineering | Separate features/target, label encoding, stratified split, scaling |
| 26 | Evaluation function | `evaluate_classifier()` — accuracy, precision, recall, F1, AUC |
| 27 | Random Forest | Train with tuned hyperparameters, evaluate |
| 28 | SVM | Train with RBF kernel, evaluate |
| 29 | XGBoost | Train with regularisation, evaluate |
| 30–33 | Comparison | Results table, bar charts, feature importance, confusion matrices |
| **34–40** | **Summary & Inference** | |
| 34–35 | Final summary | Combined results for both parts |
| 36–40 | Prediction functions | `predict_image_class()`, sample predictions, correct vs incorrect examples |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| CNN training curves | Accuracy and loss across epochs |
| CNN confusion matrix | 10-class heatmap |
| Tabular results table | RF, SVM, XGBoost metrics side-by-side |
| Tabular confusion matrices | 3-panel binary classification heatmaps |
| Feature importance chart | XGBoost ranking of all 14 Adult Income features |
| Prediction visualisations | CIFAR-10 test images with labels and confidence |

---

## Tech Stack

| Library | Role |
|---------|------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | CNN architecture, data augmentation, training |
| [scikit-learn](https://scikit-learn.org/) | Random Forest, SVM, preprocessing, evaluation metrics |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient-boosted tree classifier |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Training curves, confusion matrices, feature importance |
| [NumPy / Pandas](https://numpy.org/) | Data manipulation |

---

## License

This project is for educational and research purposes.
