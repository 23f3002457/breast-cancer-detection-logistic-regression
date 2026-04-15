import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score
)
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import kagglehub
from kagglehub import KaggleDatasetAdapter


# Load dataset
file_path = "data.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uciml/breast-cancer-wisconsin-data",
    file_path,
)

# Clean data (removing unnecessary columns)
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Convert labels (malignant = 1, benign = 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Checking class distribution to see imbalance
print("\nClass Distribution:\n")
print(y.value_counts(normalize=True))

# Train-test split (random_state for reproducibility 
# which is seed value that ensures the same shuffling sequnce every time
#  the code is run), stratifying to maintain class distribution 
# in both train and test sets (stratify=y ensures that the proportion 
# of classes in the train and test sets is similar to the original dataset).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) # Apply scaling to 
#training data this takes the mean and std of the training
#  data and applies the transformation by subtracting the mean 
# and dividing by the std to standardize the features.
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Probabilities (taking all rows and the second column which 
# corresponds to the y=1 class)
y_prob = model.predict_proba(X_test)[:, 1]

# Default prediction (threshold = 0.5)
y_pred = (y_prob > 0.5).astype(int)

# Threshold tuning
thresholds = np.linspace(0, 1, 100)

best_threshold = None
# recall is the ability of the model to correctly identify 
# positive cases (malignant tumors in this case). Recall=TP/(TP+FN) 
# where TP is true positives and FN is false negatives.
best_recall = 0

print("\nThreshold tuning:\n")

for t in thresholds:
    y_pred_t = (y_prob > t).astype(int)

    recall = recall_score(y_test, y_pred_t)
    precision = precision_score(y_test, y_pred_t)

    print(f"Threshold: {t:.2f} | Recall: {recall:.3f} | Precision: {precision:.3f}")

    if precision >= 0.90 and recall > best_recall:
        best_recall = recall
        best_threshold = t

# fixing case where no threshold satisfies condition
if best_threshold is None:
    print("No valid threshold found, using default 0.5")
    best_threshold = 0.5

print("Best threshold (balanced):", best_threshold)


y_pred_final = (y_prob > best_threshold).astype(int)

print("\nFinal Model Evaluation:\n")
print("Classification Report:\n", classification_report(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("Accuracy Score:\n", accuracy_score(y_test, y_pred_final))


wrong_idx = np.where(y_pred_final != y_test)[0]

print("\nWrong predictions count:", len(wrong_idx))
print("Wrong prediction probabilities:", y_prob[wrong_idx])

# Convert test data back to DataFrame for interpretability
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Extract wrongly predicted samples
df_wrong = X_test_df.iloc[wrong_idx]

print("\nSample wrong predictions (features):\n")
print(df_wrong.head())

# Feature importance Logistic regression coefficients
#  indicate the importance of each feature in predicting 
# the target variable.
importance = pd.Series(model.coef_[0], index=X.columns)

importance = importance.sort_values(key=abs, ascending=False)

print("\nFeature Importance:\n")
print(importance)

# converting coefficients to odds ratio (exp(coef)) gives multiplicative effect
# on odds of cancer, more interpretable than raw coefficients
odds_ratio = np.exp(importance)

print("\nOdds Ratios:\n")
print(odds_ratio)


# TOP FEATURE MODEL COMPARISON

# Select top features
top_features = importance.head(10).index
X_top = X[top_features]

# Train-test split
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler_top = StandardScaler()
X_train_top = scaler_top.fit_transform(X_train_top)
X_test_top = scaler_top.transform(X_test_top)

# Train model
model_top = LogisticRegression(max_iter=10000)
model_top.fit(X_train_top, y_train_top)

# Predict probabilities
y_prob_top = model_top.predict_proba(X_test_top)[:, 1]

# Final prediction
y_pred_top = (y_prob_top > best_threshold).astype(int)

# Compare results
print("\n=== FULL FEATURE MODEL ===")
print(classification_report(y_test, y_pred_final))

print("\n=== TOP FEATURE MODEL ===")
print(classification_report(y_test_top, y_pred_top))


# ROC Curve and AUC, which evaluates the model's ability to 
# distinguish between classes across all thresholds.
# ROC curve shows performance at different thresholds
# AUC summarizes overall performance (higher is better)

fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random model line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Curve is between recall and false positive rate at different thresholds.


# Precision Recall curve (more useful in imbalanced / medical problems)
# shows tradeoff between precision and recall across thresholds

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

plt.figure()
plt.plot(recall_vals, precision_vals, label=f"PR Curve (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


# Cross validation to check stability of model across different splits
cv_scores = cross_val_score(
    LogisticRegression(max_iter=10000),
    X, y,
    cv=5,
    scoring='recall'
)

print("Cross validation recall scores:", cv_scores)
print("Mean recall:", cv_scores.mean())


# Comparing with Random Forest (non-linear model)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr, tpr, label=f"LogReg AUC={roc_auc:.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest AUC={auc_rf:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison")
plt.legend()
plt.show()

joblib.dump(model, "model.pkl")   # learned weights
joblib.dump(scaler, "scaler.pkl") # learned transformation