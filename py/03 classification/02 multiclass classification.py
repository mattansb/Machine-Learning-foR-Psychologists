import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    make_scorer,
    RocCurveDisplay,
)
from sklearn import set_config

from palmerpenguins import load_penguins

set_config(display="diagram")

# This script demonstrates how we might assess a multiclass-classification
# model.

# Data and problem ----------------------------------------------------------

penguins = load_penguins()
# This data set contains info on penguins from the Palmer Archipelago,
# Antarctica. We will predict the species of penguins based on their bill length
# and depth, using a KNN model.

features = ["body_mass_g", "sex"]
outcome = "species"

X = penguins[features]
y = penguins[outcome]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=20251201, stratify=y
)
# Note: stratify ensures both train and test have similar class distributions


# Tune ---------------------------------------------------------------------

# We will be tuning the number of neighbors (k) in the KNN model.
knn_spec = KNeighborsClassifier()

# The data contains missing values - we will impute using the median / mode:
# We need to dummy code the factor predictors for KNN
# We need to normalize the predictors for KNN

cat_cols = ["sex"]
num_cols = ["body_mass_g"]

col_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(drop="first", sparse_output=False),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]
)

preprocessor = Pipeline(
    steps=[
        ("imputer", col_preprocessor),
        ("scaler", StandardScaler()),
    ]
)

knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", knn_spec)]
)

# Unlike binary classification, we don't typically have a "positive" class, so
# we can't really compute a single sensitivity or specificity (etc.) value.
# Instead, we can use several methods to summarize the performance of a
# multiclass model - with the default being the "macro" method, which simply
# computes the metric for each class, and then averages them.
#
# Note, however that accuracy does not require a "positive" class, and so it can
# be used without issue in multiclass problems.


def sensitivity_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")


def specificity_macro(y_true, y_pred):
    # Specificity needs to be calculated per class, then averaged
    # We'll compute it as the recall of the "negative" class for each one-vs-rest
    cm = confusion_matrix(y_true, y_pred)
    # Specificity for each class
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return np.mean(specificity_per_class)


scoring = {
    "sensitivity_macro": make_scorer(sensitivity_macro),
    "specificity_macro": make_scorer(specificity_macro),
    "accuracy": make_scorer(accuracy_score),
}

# Setup the tuning grid:
param_grid = {"classifier__n_neighbors": [5, 36, 68, 100]}

# Setup 10-fold CV:
cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=20251201)

# Perform the tuning:
tune_results = GridSearchCV(
    knn_pipeline,
    param_grid=param_grid,
    cv=cv_folds,
    scoring=scoring,
    refit="accuracy",
    n_jobs=-1,
)

tune_results.fit(X_train, y_train)

# Visualize tuning results
results_df = pd.DataFrame(tune_results.cv_results_)
results_long = results_df.melt(
    id_vars=["param_classifier__n_neighbors"],
    value_vars=[
        "mean_test_sensitivity_macro",
        "mean_test_specificity_macro",
        "mean_test_accuracy",
    ],
    var_name="metric",
    value_name="value",
)

# Clean up metric names
results_long["metric"] = results_long["metric"].str.replace("mean_test_", "")

p_tune = (
    ggplot(
        results_long,
        aes(x="param_classifier__n_neighbors", y="value", color="metric"),
    )
    + geom_line(size=1)
    + geom_point(size=3)
    + labs(
        x="K (neighbors)",
        y="Metric Value",
        color="Metric",
        title="Tuning Results",
    )
    + theme_minimal()
    + theme(figure_size=(10, 6))
)
p_tune.draw(show=True)

## Finalize the model ----------------------------------------------------

# What's the best K?
best_k = tune_results.best_params_["classifier__n_neighbors"]
print(f"\nBest K (by accuracy): {best_k}")
# (What would the one-SE rule pick?)

knn_fit = tune_results.best_estimator_


# Predict -----------------------------------------------------------------

y_pred_class = knn_fit.predict(X_test)
y_pred_proba = knn_fit.predict_proba(X_test)

print("\nFirst rows of predictions:")
print(y_pred_class[:5])
print(y_pred_proba[:5, :])

# scikit-learn provides 3 averaging methods for dealing with multiclass
# predictions:
# - macro: compute metric for each class, then average (unweighted)
# - weighted: compute metric for each class, then average weighted by class
#   frequency
# - micro: compute metric globally by counting total true positives, false
#   negatives, etc.
# You can read about these here:
# https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification

## Macro (standard averaging) (default)
sensitivity_macro_test = recall_score(y_test, y_pred_class, average="macro")
specificity_macro_test = specificity_macro(y_test, y_pred_class)
accuracy_test = accuracy_score(y_test, y_pred_class)

print("\n" + "=" * 60)
print("Test Set Performance (Macro averaging):")
print("=" * 60)
print(f"Sensitivity: {sensitivity_macro_test:.3f}")
print(f"Specificity: {specificity_macro_test:.3f}")
print(f"Accuracy:    {accuracy_test:.3f}")

## Weighted Macro (weight by class frequency)
sensitivity_weighted = recall_score(y_test, y_pred_class, average="weighted")
print(f"\nSensitivity (weighted): {sensitivity_weighted:.3f}")

## Micro (sort of observation wise averaging)
sensitivity_micro = recall_score(y_test, y_pred_class, average="micro")
print(f"Sensitivity (micro):    {sensitivity_micro:.3f}")


# Event-wise metrics (not averaged)
sensitivity_classwise = recall_score(y_test, y_pred_class, average=None)
print("\n" + "=" * 60)
print("Per-Class Metrics:")
print("=" * 60)

classes = knn_fit.classes_

for cls in range(len(classes)):
    print(f"  Sensitivity {classes[cls]}: {sensitivity_classwise[cls]:.3f}")

# We can see we have great sensitivity and specificity Gentoo, while having poor
# sensitivity for Chinstrap / specificity for Adelie. This is because the model
# is very good at predicting Gentoo, but not so good at predicting the other
# classes.

# We can get a sense for this using a ROC curve:
# For multiclass, we compute ROC for each class vs rest

# We can get a sense for this using a ROC curve:
y_test_onehot = LabelBinarizer().fit_transform(y_test)

fig, ax = plt.subplots(figsize=(6, 6))
for class_id in range(3):
    RocCurveDisplay.from_predictions(
        y_test_onehot[:, class_id],
        y_pred_proba[:, class_id],
        name=classes[class_id],
        # curve_kwargs=dict(color=color),
        ax=ax,
        plot_chance_level=(class_id == 2),
    )
fig.show()

# We can get an "average" AUC:
auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
print(f"AUC (Average) = {auc:.3f}")
