import numpy as np
import pandas as pd
from plotnine import *

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
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


# Define metrics
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["No", "Yes"]
    ).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def j_index_score(y_true, y_pred):
    # Youden's J statistic = sensitivity + specificity - 1
    sens = recall_score(y_true, y_pred, pos_label="Impaired")
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

list(y_pred_class_50).index("Yes")
# get unique values from list
impaired_idx = list(knn_fit.classes_).index("Impaired")
classes = knn_fit.classes_

# Calculate metrics
sensitivity_default = recall_score(y_test, y_pred_class_50, pos_label="Yes")
specificity_default = specificity_score(y_test, y_pred_class_50)
accuracy_default = accuracy_score(y_test, y_pred_class_50)
j_index_default = j_index_score(y_test, y_pred_class_50)
auc_default = roc_auc_score(y_test, y_pred_proba_50[:, impaired_idx])

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

# To change the threshold, we need to tune it as a hyperparameter.
# TODO fix me
# Python/scikit-learn doesn't have a direct equivalent to the {tailor} package
# We'll implement threshold adjustment manually

# There are calibration methods available:
# https://scikit-learn.org/stable/modules/calibration.html
# But for threshold selection, we'll implement a custom approach


# Custom threshold selection approach
# We'll evaluate different thresholds using cross-validation


class ThresholdClassifier:
    """Wrapper that applies a custom threshold to probabilistic predictions"""

    def __init__(self, estimator, threshold=0.5, pos_label_idx=0):
        self.estimator = estimator
        self.threshold = threshold
        self.pos_label_idx = pos_label_idx

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        # Apply threshold
        predictions = np.where(
            proba[:, self.pos_label_idx] >= self.threshold,
            self.classes_[self.pos_label_idx],
            self.classes_[1 - self.pos_label_idx],
        )
        return predictions

    def set_params(self, **params):
        # Handle threshold parameter
        if "threshold" in params:
            self.threshold = params.pop("threshold")
        # Pass remaining params to estimator
        if params:
            self.estimator.set_params(**params)
        return self

    def get_params(self, deep=True):
        params = {
            "threshold": self.threshold,
            "pos_label_idx": self.pos_label_idx,
        }
        if deep:
            params.update(
                {
                    f"estimator__{k}": v
                    for k, v in self.estimator.get_params(deep=True).items()
                }
            )
        return params


## Tune the threshold -------------------------------------------------

# Create a threshold-adjustable classifier
threshold_pipeline = ThresholdClassifier(
    knn_pipeline, pos_label_idx=impaired_idx
)

# Define scoring metrics
scoring = {
    "sensitivity": make_scorer(
        recall_score, pos_label="Impaired", zero_division=0
    ),
    "specificity": make_scorer(specificity_score, zero_division=0),
    "accuracy": make_scorer(accuracy_score),
    "j_index": make_scorer(j_index_score),
    "roc_auc": make_scorer(
        lambda y_true, y_pred: roc_auc_score(
            y_true, threshold_pipeline.predict_proba(X_train)[:, impaired_idx]
        )
    ),
}

# Setup tuning grid for threshold
param_grid = {"threshold": np.linspace(0, 1, 20)}

# Setup CV
cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20251201)

# Tune
tune_results = GridSearchCV(
    threshold_pipeline,
    param_grid=param_grid,
    cv=cv_folds,
    scoring=scoring,
    refit="j_index",  # Select best threshold based on J-index
    n_jobs=-1,
)

tune_results.fit(X_train, y_train)

# Visualize tuning results
results_df = pd.DataFrame(tune_results.cv_results_)
results_long = results_df.melt(
    id_vars=["param_threshold"],
    value_vars=[
        "mean_test_sensitivity",
        "mean_test_specificity",
        "mean_test_j_index",
    ],
    var_name="metric",
    value_name="value",
)

results_long["metric"] = results_long["metric"].str.replace("mean_test_", "")

p_tune1 = (
    ggplot(results_long, aes(x="param_threshold", y="value", color="metric"))
    + geom_line(size=1)
    + labs(
        x="Threshold",
        y="Metric Value",
        color="Metric",
        title="Tuning Results: Sensitivity, Specificity, and J-index",
    )
    + theme_minimal()
    + theme(figure_size=(10, 6))
)
p_tune1.draw(show=True)

# As expected, we can see that sensitivity and specificity trade off from 0/1 to
# 1/0 as we change the threshold, and J-index seems to hit a good balance around 0.3.

# Accuracy is also affected by threshold changes - but AUC is *not*. Why?
results_long2 = results_df.melt(
    id_vars=["param_threshold"],
    value_vars=["mean_test_accuracy", "mean_test_roc_auc"],
    var_name="metric",
    value_name="value",
)
results_long2["metric"] = results_long2["metric"].str.replace("mean_test_", "")

p_tune2 = (
    ggplot(results_long2, aes(x="param_threshold", y="value", color="metric"))
    + geom_line(size=1)
    + labs(
        x="Threshold", y="Metric Value", color="Metric", title="Accuracy vs AUC"
    )
    + theme_minimal()
    + theme(figure_size=(10, 6))
)
p_tune2.draw(show=True)

## Finalize model ------------------------------------------------

best_threshold = tune_results.best_params_["threshold"]
print(f"\nBest threshold (by J-index): {best_threshold:.3f}")

# We can also select by optimizing multiple metrics simultaneously
# TODO fix me
# Python doesn't have a direct equivalent to the {desirability} package
# We would need to implement custom multi-objective optimization

# Get the best estimator
knn_fit_adjusted = tune_results.best_estimator_

# Predict (+ adjust) on test set ---------------------------------

y_pred_class_adjusted = knn_fit_adjusted.predict(X_test)
y_pred_proba_adjusted = knn_fit_adjusted.predict_proba(X_test)

ad_test_pred_adjusted = pd.DataFrame(
    {
        "Class": y_test.values,
        "pred_class": y_pred_class_adjusted,
        "pred_Impaired": y_pred_proba_adjusted[:, impaired_idx],
    }
)

# We can see that the class predictions do not completely agree:
comparison = pd.crosstab(
    ad_test_pred_default["pred_class"],
    ad_test_pred_adjusted["pred_class"],
    rownames=["Default (0.50)"],
    colnames=["Adjusted (thresh)"],
)
print("\nPrediction comparison:")
print(comparison)

# (But note that probabilistic predictions are unaffected by adjustment)
print("\nProbabilistic predictions comparison:")
print(
    f"Correlation: {np.corrcoef(ad_test_pred_default['pred_Impaired'], ad_test_pred_adjusted['pred_Impaired'])[0, 1]:.4f}"
)

# Plot comparison
comparison_df = pd.DataFrame(
    {
        "default": ad_test_pred_default["pred_Impaired"],
        "adjusted": ad_test_pred_adjusted["pred_Impaired"],
    }
)

p_compare = (
    ggplot(comparison_df, aes(x="default", y="adjusted"))
    + geom_point(alpha=0.5)
    + geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
    + labs(
        x="Default (thresh = 0.50)",
        y=f"Adjusted (thresh = {best_threshold:.2f})",
        title="Probabilistic Predictions Comparison",
    )
    + theme_minimal()
    + coord_equal()
)
p_compare.draw(show=True)

# Calculate metrics
sensitivity_adjusted = recall_score(
    y_test, y_pred_class_adjusted, pos_label="Impaired"
)
specificity_adjusted = specificity_score(y_test, y_pred_class_adjusted)
accuracy_adjusted = accuracy_score(y_test, y_pred_class_adjusted)
j_index_adjusted = j_index_score(y_test, y_pred_class_adjusted)
auc_adjusted = roc_auc_score(y_test, y_pred_proba_adjusted[:, impaired_idx])

print("\n" + "=" * 60)
print("Test Set Performance Comparison:")
print("=" * 60)
print(f"{'Metric':<15} {'Default (0.50)':<20} Adjusted ({best_threshold:.2f})")
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
