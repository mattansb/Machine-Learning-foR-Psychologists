import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

from palmerpenguins.penguins import load_penguins

# This script demonstrates how we might asses a multiclass-classification model.

# Data and problem ----------------------------------------------------------

penguins = load_penguins()
penguins["species"] = pd.Categorical(penguins["species"])
penguins.head()
# This data set contains info on penguins from the Palmer Archipelago,
# Antarctica. We will predict the species of penguins based on their bill length
# and depth, using a KNN model.

outcome = "species"
features = ["body_mass_g", "sex"]

X_train, X_test, y_train, y_test = train_test_split(
    penguins[features], penguins[outcome], train_size=0.7, random_state=1234
)


# Fit ---------------------------------------------------------------------

preprocessor = Pipeline(
    steps=[
        (
            "ct1",
            ColumnTransformer(
                transformers=[
                    (
                        "imp_med_mass",
                        SimpleImputer(strategy="median"),
                        ["body_mass_g"],
                    ),
                    (
                        "imp_mode_sex",
                        SimpleImputer(strategy="most_frequent"),
                        ["sex"],
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        (
            "ct2",
            ColumnTransformer(
                transformers=[
                    # column has been renamed after ct1
                    (
                        "dummy",
                        OneHotEncoder(drop="first"),
                        ["imp_mode_sex__sex"],
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        ("z", StandardScaler()),  # everything
    ]
)

clf = KNeighborsClassifier(n_neighbors=5)

knn_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifer", clf)])

knn_pipe.fit(X_train, y_train)


# Predict -----------------------------------------------------------------

y_pred = knn_pipe.predict(X_test)

y_pred_prob = knn_pipe.predict_proba(X_test)

# Unlike binary classification, we don't typically have a "positive" class, so
# we can't really compute sensitivity, specificity, etc. Instead, we can use
# several methods to summarize the performance of a multiclass model.
#
# Note, however that accuracy does not require a "positive" class, and so it can
# be used without issue in multiclass problems.
accuracy_score(y_test, y_pred)


# sklearn.metrics provids 3 methods for dealing with multiclass predictions, all
# of them effectively compute such metrics for each class, and them average
# them.
recall_score(y_test, y_pred, average="macro")  # standard averaging
recall_score(y_test, y_pred, average="weighted")  # weighted by class frequency
recall_score(
    y_test, y_pred, average="micro"
)  # sort of observation wise averaging

# We can also get these metrics by event:
recall_score(y_test, y_pred, average=None)
# We can see we have great sensitivity for Gentoo, while having poor sensitivity
# for Chinstrap (and what's going on with Adelie?). This is because the model
# is very good at predicting Gentoo, but not so good at predicting the other
# classes.

# We can get a sense for this using a ROC curve:
y_test_onehot = LabelBinarizer().fit_transform(y_test)
colors = ["aqua", "darkorange", "cornflowerblue"]

fig, ax = plt.subplots(figsize=(6, 6))
for class_id, color in zip(range(3), colors):
    RocCurveDisplay.from_predictions(
        y_test_onehot[:, class_id],
        y_pred_prob[:, class_id],
        name=y_test.cat.categories[class_id],
        curve_kwargs=dict(color=color),
        ax=ax,
        plot_chance_level=(class_id == 2),
    )


# We can get an "average" AUC:
roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
