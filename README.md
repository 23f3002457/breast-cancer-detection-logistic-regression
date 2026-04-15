# breast-cancer-detection-logistic-regression
# Breast Cancer Detection (ML)

This project builds a machine learning model to classify tumors as malignant or benign using the Breast Cancer Wisconsin dataset.

## Overview

* Binary classification problem (Malignant = 1, Benign = 0)
* Logistic Regression used as baseline model
* Focus on medical constraint: **high recall while maintaining good precision**

## Dataset

Breast Cancer Wisconsin Dataset (UCI / Kaggle version)
Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Approach

* Data cleaning (removed unnecessary columns)
* Feature scaling using StandardScaler
* Train-test split with stratification to maintain class balance
* Logistic Regression model training
* Probability-based predictions instead of direct labels

## Threshold Tuning

Instead of using default threshold (0.5), thresholds are tested from 0 to 1.

Goal:

* Maintain **precision ≥ 0.90**
* Maximize **recall**

This is important because missing cancer cases (false negatives) is more critical.

## Evaluation

* Classification Report
* Confusion Matrix
* Accuracy (for reference)
* ROC Curve (AUC)
* Precision-Recall Curve

## Error Analysis

* Identified wrongly predicted samples
* Checked prediction probabilities for misclassified points
* Helps understand where model is uncertain or making mistakes

## Feature Importance

* Logistic regression coefficients used
* Converted to **odds ratios** for better interpretability

## Model Comparison

* Compared full feature model vs top 10 features
* Observed minimal drop in recall → indicates feature redundancy

## Additional

* Cross-validation used to check model stability
* Random Forest model used for comparison with logistic regression

## Notes

* Model is trained on scaled data, so scaler is saved along with model
* Predictions require same preprocessing (scaling)

## Future Work

* Improve model with non-linear methods
* Better feature engineering
* Deployment as simple prediction tool

---
