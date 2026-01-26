# ### Tutorial 6- Clustering  ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import set_config
from helpers import *
from palmerpenguins import load_penguins

# Configure sklearn to output pandas DataFrames
set_config(transform_output="pandas")

# Learn more at:
# https://scikit-learn.org/stable/modules/clustering.html

# The Data ---------------------------------------

penguins = load_penguins()
penguins.species = penguins.species.astype("category")
penguins.info()

# (Ignoring "species" - ) Are there clusters of penguins that are similar in
# their measurements?

## Prep -------------------

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

penguins.dropna(subset=features, inplace=True)

print(penguins.loc[:, features].describe())
# variables are on very different scales.
#
# We should re-scale them.

# We can use StandardScaler: # ! NEW PY
penguins_z = pd.DataFrame(
    StandardScaler().fit_transform(penguins.loc[:, features]), columns=features
)

print(penguins_z.head())

sns.pairplot(penguins_z)
plt.show()
# there are some associations variables, but are there CLUSTERS?


## t-SNE plot -------------------------------------------------

# t-SNE plots are useful for visualizing high dimensional data in 2D space.
# Unlike PCA or other methods we've discussed, t-SNE is *only* good for
# visualization - it is a non-linear method the preserve the local structure of
# data but not the global structure of the data.
# Read more:
# https://biostatsquid.com/pca-umap-tsne-comparison/
# https://pair-code.github.io/understanding-umap/

# It is also non-deterministic = it will return different results each time,
# so don't forget to set a seed (random_state):
tsne = TSNE(n_components=2, perplexity=5, random_state=20260126)
# Default perplexity is 30, but this value is too large for our small dataset.

penguins_z_tSNE = tsne.fit_transform(penguins_z)

p_tSNE = (
    ggplot(penguins_z_tSNE, aes("tsne0", "tsne1"))
    + geom_point(size=2)
    + theme_void()
)
p_tSNE.show()
# It seems like there are 3 or 4 clumps of high-D (4D in out case) data.


# Partitioning Clustering --------------------------------------------------------

# Partitioning clustering tries to classify observations into mutually exclusive
# clusters, such that observations within the same cluster are as similar as
# possible (i.e., high intra-class similarity), whereas observations from
# different clusters are as dissimilar as possible (i.e., low inter-class
# similarity)

# This is a "Top Down" approach - cluster observations on the basis of the
# features. Prior selection of K - number of clusters.

## Choosing number of clusters ---------------------
# https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/


def plot_metric_by_k(metric, ylabel=None, cutoff=None):
    K = np.array(range(len(metric))) + 1

    plt.figure(figsize=(8, 4))
    plt.plot(K, metric, "bx-")
    plt.xlabel("k")
    if ylabel is not None:
        plt.ylabel(ylabel)
    if cutoff is not None:
        plt.axvline(x=cutoff, linestyle="--", color="k")
    plt.show()


K = range(1, 20)

## Elbow method: drop in within-cluster variance:
inertias = []
for k in K:
    km = KMeans(n_clusters=k, random_state=20260125).fit(penguins_z)
    inertias.append(km.inertia_)
plot_metric_by_k(inertias, ylabel="SSW", cutoff=2)
# Note that, the elbow method is sometimes ambiguous.

## Average silhouette method: Measures the quality of a clustering - how well
# each object lies within its cluster. A high average silhouette width indicates
# a good clustering.
silhouette_scores = [0]  # silhouette is undefined for k=1
for k in K:
    if k == 1:
        continue
    km = KMeans(n_clusters=k, random_state=20260125).fit(penguins_z)
    silhouette_scores.append(silhouette_score(penguins_z, km.labels_))
plot_metric_by_k(silhouette_scores, ylabel="Silhouette Score", cutoff=2)
# Here, k=2

## Gap statistic (using bootstrap sampling): measures how far is the observed
# within intra-cluster variation is from a random uniform distribution of the
# data?
gap_values = compute_gap_statistic(penguins_z.values, k_max=20, n_replicates=10)
plot_metric_by_k(gap_values, ylabel="Gap Statistic", cutoff=3)
# Suggests k=6!

## Finding Clusters with k-means ------------------------------

# The KMeans class performs K-means clustering.
km = KMeans(
    n_clusters=2,  # K
    n_init=20,  # how many random starting centers
    random_state=20260126,
)
km.fit(penguins_z)
# Note: If n_init > 1, then K-means clustering will be performed using multiple
# random assignments in Step 1 of the Algorithm, and KMeans will report only the
# best results.
# It is recommended always running K-means clustering with a large start, such
# as 20 or 50, since otherwise an undesirable local optimum may be obtained

print(km.labels_)  # vector which assigns obs. to each cluster
print(pd.Series(km.labels_).value_counts())  # obs. num in each cluster

print(km.cluster_centers_)  # the two clusters' centers
# We can also find the centers on the original scale:
print(penguins.loc[:, features].groupby(km.labels_).mean())

cluster_km = pd.Categorical(km.labels_)

# Plotting each observation colored according to its cluster assignment:
# Using seaborn for scatterplot matrix with hue
penguins_plot = penguins.copy().loc[:, features]
penguins_plot["cluster"] = cluster_km
sns.pairplot(penguins_plot, hue="cluster")
plt.show()

# How does this look on our t-SNE plot?
(p_tSNE + aes(color=cluster_km)).show()
