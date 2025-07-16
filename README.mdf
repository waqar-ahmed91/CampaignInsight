
# CampaignInsight

**AUC-Optimized Mail Response Prediction Pipeline Using XGBoost and Class Imbalance Correction**

## Overview

**CampaignInsight** is a production-ready machine learning pipeline designed to predict customer responses in direct mail campaigns. The project focuses on rare event classification, addressing class imbalance through robust preprocessing, AUC-optimized model training, and threshold tuning for high-precision decision making.

This pipeline is ideal for marketing teams, data scientists, or ML engineers looking to prioritize outreach efforts and maximize ROI by accurately identifying potential responders in large-scale mailing lists.

---

## Features

- **End-to-End Preprocessing**  
  Handles missing data, removes low-information features, and encodes categorical variables.

- **Class Imbalance Handling**  
  Supports both `SMOTE` oversampling and `scale_pos_weight` via XGBoost for rare positive classes.

- **AUC-Focused Training & Thresholding**  
  Trains models using `XGBoostClassifier` with hyperparameter optimization and threshold sweeping to maximize ROC AUC and PR AUC scores.

- **Validation & Hold-out Testing**  
  Clean separation of training, validation, and final test datasets for true generalization evaluation.

- **Modular and Scalable Codebase**  
  Designed for datasets with hundreds of features and millions of rows. Easily adaptable to new domains or business rules.

---

## Installation

```bash
git clone https://github.com/yourusername/CampaignInsight.git
cd CampaignInsight
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- scikit-learn
- xgboost
- imbalanced-learn
- pandas, numpy, tqdm

---

## Quick Start

### 1. Load and Preprocess Your Dataset

```python
from src.preprocessing import full_preprocessing_pipeline_with_validation

X_train, y_train, X_val, y_val, preprocessor = full_preprocessing_pipeline_with_validation(
    df=your_dataframe,
    label_col='RESPONSE',
    id_columns=['LNR'],
    sample_fraction=1.0,
    val_size=0.2,
    stratify=True,
    balance_strategy='smote',
    outlier_removal=False,
    random_state=42
)
```

### 2. Train and Tune Your Model

```python
from src.training import train_and_evaluate_model_auc

model, preprocessor, best_thresh = train_and_evaluate_model_auc(
    df_trainval,
    label_col='RESPONSE',
    id_columns=['LNR'],
    null_threshold=0.9,
    test_size=0.2,
    smote=True,
    search=True,
    threshold=None,
    use_gpu=True,
    n_jobs=2
)
```

### 3. Evaluate on Unseen Test Data

```python
from src.evaluation import evaluate_on_test

evaluate_on_test(
    df_test,
    model=model,
    preprocessor=preprocessor,
    threshold=best_thresh,
    label_col='RESPONSE',
    id_columns=['LNR']
)
```

---

## Results

The pipeline consistently achieved:

- **ROC AUC**: ~0.84–0.85 on unseen test sets  
- **PR AUC**: ~0.34–0.36 on highly imbalanced data  
- **Low False Positive Rates** with threshold tuning  
- **High Precision** in positive class predictions  

These results demonstrate strong generalization for rare event prediction in mailout response scenarios.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or collaboration inquiries, please reach out via [GitHub Issues](https://github.com/waqar-ahmed91/CampaignInsight/issues).
