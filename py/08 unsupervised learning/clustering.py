# ### Tutorial 6- Clustering  ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import load_wine
from sklearn import set_config
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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


# Hierarchical Clustering ---------------------------------------------------------

# This is a "Bottom Up" approach- cluster features on the basis of the
# observations. NO prior selection of num. of clusters.

# We will use the same data to plot the hierarchical clustering dendrogram using
# complete, single, and average linkage clustering, with Euclidean distance as
# the dissimilarity measure.

## Distance metric -----------------------------------
# There are several to choose from....
# ! NEW PY: scipy.spatial.distance.pdist

penguins_d_euc = pdist(penguins_z, metric="euclidean")
penguins_d_cor = pdist(penguins_z, metric="correlation")
# and many more...
help(pdist)


# Which should we use?
def plot_distance_matrix(d_matrix):
    d_square = squareform(d_matrix)
    plt.figure(figsize=(6, 5))
    sns.heatmap(d_square, cmap="viridis")
    plt.title("Distance Matrix")
    plt.show()


plot_distance_matrix(penguins_d_euc)
plot_distance_matrix(penguins_d_cor)


## Build dendrogram -----------------
# The linkage() function implements hierarchical clustering.

hc_complete = linkage(penguins_d_euc, method="complete")
# Or method='average'
# Or method='single'
# Or method='centroid'

# plotting the dendrograms (with small-ish samples) and determining clusters:
plt.figure(figsize=(10, 5))
dendrogram(hc_complete)
plt.show()


## Cut the tree! -----------------------------------

# To determine the cluster labels for each observation associated with a given
# cut of the dendrogram, we can use the fcluster() function:
hc_cut_k6 = fcluster(hc_complete, t=6, criterion="maxclust")
hc_cut_h5 = fcluster(hc_complete, t=5, criterion="distance")

# Plotting each observation colored according to its cluster assignment:
penguins_plot["hclust"] = pd.Categorical(hc_cut_h5)
sns.pairplot(penguins_plot, hue="hclust")
plt.show()

# How does this look on our t-SNE plot?
(p_tSNE + aes(color=penguins_plot["hclust"])).show()


# Do the methods agree?
print(pd.crosstab(hc_cut_h5, cluster_km))


# Model based clustering -----------------------------------
#
# See sklearn.mixture.GaussianMixture
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

# Understanding Clusters ------------------------------------------------------
# All clustering methods will produce clusters, even if there is no real
# structure in the data. So, it is important to validate the clusters.

## Internal Validation? -------------------------------------------------
# How "good" is the clustering?
# There are several metrics to evaluate clustering quality on the training data.

### Silhouette width ----------------------------
# Silhouette width measures how similar an observation is to its own cluster
# compared to other clusters.

sil_values = silhouette_samples(penguins_z, cluster_km)
sil_data = pd.DataFrame({"silhouette_width": sil_values, "cluster": cluster_km})
(
    ggplot(sil_data, aes("silhouette_width", fill="cluster"))
    + geom_histogram(alpha=0.6, position="identity")
)

print("Average Silhouette Score:")
print(pd.Series(sil_values).groupby(cluster_km).mean())
# Overall the values are adequate, but not amazing...

### Cluster stability ----------------------------
# Are the clusters stable to small perturbations in the data? In other words,
# if we re-sample the data, will we get similar clusters?

# ! cluster stability analysis not imlpamented in python :(


### Compare cluster means ---------------
# Do the clusters differ on the variables that went into the clustering?

# ! selective inference not imlpamented in python :(


## External Validation? -------------------------------------------------
# Are these clusters *useful*?
# So far we've seen how the clusters map *back onto* the variables that went
# into the clustering - but these results are trivial. The question is can these
# clusters be used to tell us something new about other variables?

# For example... how about species in the penguin data?
print(pd.crosstab(penguins.species, cluster_km))
# Not perfect, but pretty good!

# Ideally we would have other variables not used in the clustering to
# validate the clusters. See R examples.


# Exercise ----------------------------------------------------------------

wine = load_wine(as_frame=True)
wine_data = wine.data
wine_data["target"] = wine.target.astype("category")
print(wine_data.info())

# 0. Build a t-SNE plot based on all columns (minus "target").
#   - Color the point by target
# 1. Cluster wines into groups based on these data
#   - use hclust
#   - decide on a distance metric
#   - choose a linkage method
#   - plot the dendrogram - and choose the number of clusters
#   - plot the clusters on a t-SNE plot.
# 2. Validate the clusters:
#   - Are the results stable?
#   - Do the clusters differ on the variables used to create them?
#   - Do the clusters map nicely onto the "target" variable?
