import numpy as np
import pandas as pd
from plotnine import *

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    TunedThresholdClassifierCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    make_scorer,
    classification_report,
)
from sklearn import set_config

from ISLP import load_data

set_config(display="diagram")


# The Data & Problem --------------------------------------------------------

Default = load_data("Default")
print(Default.info())

# Data Splitting
outcome = "default"
features = [col for col in Default.columns if col != outcome]


X_train, X_test, y_train, y_test = train_test_split(
    Default[features],
    Default[outcome],
    train_size=0.7,
    random_state=20251202,
    stratify=Default[outcome],
)

# Check class distribution
print(y_train.value_counts(normalize=True))


# Define metrics
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["No", "Yes"]
    ).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def j_index_score(y_true, y_pred):
    # Youden's J statistic = sensitivity + specificity - 1
    sens = recall_score(y_true, y_pred, pos_label="Yes")
    spec = specificity_score(y_true, y_pred)
    return sens + spec - 1


# Fit a model -------------------------------------------------------------

# Preprocessing Spec
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            ["balance", "income"],
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "encoder",
                        OneHotEncoder(drop="first", sparse_output=False),
                    ),
                    ("z", StandardScaler()),
                ]
            ),
            ["student"],
        ),
    ]
)

# Model Spec
knn_spec = KNeighborsClassifier(n_neighbors=15)  # arbitrary choice

knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", knn_spec)]
)

knn_fit = knn_pipeline.fit(X_train, y_train)


## Evaluate on hold-out set ---------------------------------

y_pred_class_50 = knn_fit.predict(X_test)
y_pred_proba_50 = knn_fit.predict_proba(X_test)

# get unique values from list
classes = knn_fit.classes_

# Calculate metrics
sensitivity_default = recall_score(y_test, y_pred_class_50, pos_label="Yes")
specificity_default = specificity_score(y_test, y_pred_class_50)
accuracy_default = accuracy_score(y_test, y_pred_class_50)
j_index_default = j_index_score(y_test, y_pred_class_50)
auc_default = roc_auc_score(y_test, y_pred_proba_50[:, 1])

print("\n" + "=" * 60)
print("Test Set Performance (threshold = 0.50):")
print("=" * 60)
print(f"Sensitivity: {sensitivity_default:.3f}")
print(f"Specificity: {specificity_default:.3f}")
print(f"Accuracy:    {accuracy_default:.3f}")
print(f"J-index:     {j_index_default:.3f}")
print(f"ROC AUC:     {auc_default:.3f}")
print("=" * 60)

# The model is pretty good, but early detection of Alzheimer's disease is
# important - we want higher sensitivity (at the cost of lower specificity).
# (Note also that the data is imbalanced, but even so...)

# What can we do?

# Find a threshold -----------------------------------------------------------

# To change the threshold, we need to tune it as a hyperparameter:
knn_tuned_tresh = TunedThresholdClassifierCV(
    knn_fit,
    # Define metrics to compute during tuning
    scoring=make_scorer(j_index_score, response_method="predict"),
    # Setup tuning grid for threshold
    thresholds=np.linspace(0, 1, 10),
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=20251201),
    n_jobs=-1,
)
knn_tuned_tresh.fit(X_train, y_train)

print(f"Cut-off point found at {knn_tuned_tresh.best_threshold_:.3f}")
print(classification_report(y_test, knn_tuned_tresh.predict(X_test)))


# Predict (+ adjust) on test set ---------------------------------

y_pred_class_adjusted = knn_tuned_tresh.predict(X_test)
y_pred_proba_adjusted = knn_tuned_tresh.predict_proba(X_test)

# We can see that the class predictions do not completely agree:
comparison = pd.crosstab(
    y_pred_class_adjusted,
    y_pred_class_50,
    rownames=["Default (0.50)"],
    colnames=["Adjusted (thresh)"],
)
print("\nPrediction comparison:")
print(comparison)

# (But note that probabilistic predictions are unaffected by adjustment)
print("\nProbabilistic predictions comparison:")
print(
    f"Correlation: {np.corrcoef(y_pred_proba_50[:, 1], y_pred_proba_adjusted[:, 1])[0, 1]:.4f}"
)

# Calculate metrics
sensitivity_adjusted = recall_score(
    y_test, y_pred_class_adjusted, pos_label="Yes"
)
specificity_adjusted = specificity_score(y_test, y_pred_class_adjusted)
accuracy_adjusted = accuracy_score(y_test, y_pred_class_adjusted)
j_index_adjusted = j_index_score(y_test, y_pred_class_adjusted)
auc_adjusted = roc_auc_score(y_test, y_pred_proba_adjusted[:, 1])

print("\n" + "=" * 60)
print("Test Set Performance Comparison:")
print("=" * 60)
print(
    f"{'Metric':<15} {'Default (0.50)':<20} Adjusted ({knn_tuned_tresh.best_threshold_:.2f})"
)
print("-" * 60)
print(
    f"{'Sensitivity':<15} {sensitivity_default:<20.3f} {sensitivity_adjusted:.3f}"
)
print(
    f"{'Specificity':<15} {specificity_default:<20.3f} {specificity_adjusted:.3f}"
)
print(f"{'Accuracy':<15} {accuracy_default:<20.3f} {accuracy_adjusted:.3f}")
print(f"{'J-index':<15} {j_index_default:<20.3f} {j_index_adjusted:.3f}")
print(f"{'ROC AUC':<15} {auc_default:<20.3f} {auc_adjusted:.3f}")
print("=" * 60)
