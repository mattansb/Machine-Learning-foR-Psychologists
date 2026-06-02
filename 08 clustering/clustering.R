### Tutorial 6- Clustering  ###

library(tidymodels)
library(tidyclust)

library(cluster)
library(philentropy)
library(Rtsne)


# Learn more at:
# https://tidyclust.tidymodels.org/index.html

# The Data ---------------------------------------

data(USArrests)
?USArrests # Violent Crime Rates by US State
USArrests <- USArrests |> tibble::rownames_to_column("state.name")
head(USArrests)

plot(USArrests[, 2:5], pch = 20, cex = 2)
# there are some associations between variables, but are there CLUSTERS?

# Note: no need to split into train/test sets for clustering, since we are not
# trying to predict anything. We will be using all the data to find clusters,
# and then we will validate the clusters using internal and external validation
# methods.

## Preprocessing -------------------

summary(USArrests)
# variables are on very different scales.
# We should re-scale them.

rec <- recipe(~ Murder + Assault + UrbanPop + Rape, data = USArrests) |>
  step_normalize(all_numeric_predictors())

# We will make a copy of the processed data:
USArrests_z <- prep(rec) |> bake(new_data = USArrests)

## t-SNE plot -------------------------------------------------

# t-SNE plots are useful for visualizing high dimensional data in 2D space.
# Unlike PCA or other methods we've discussed, t-SNE is *only* good for
# visualization - it is a non-linear method the preserve the local structure of
# data but not the global structure of the data.
# Read more:
# https://biostatsquid.com/pca-umap-tsne-comparison/
# https://pair-code.github.io/understanding-umap/

# It is also non-deterministic = it will return different results each time,
# so don't forget to set a seed:
set.seed(20251109)

USArrests_tSNE <- Rtsne(USArrests_z, perplexity = 5, normalize = FALSE)
# Default perplexity is 30, but this value is too large for our small dataset.

p_tSNE <- data.frame(USArrests_tSNE$Y) |>
  ggplot(aes(X1, X2)) +
  geom_point(size = 2) +
  ggrepel::geom_text_repel(aes(label = USArrests$state.name)) +
  # scales are meaningless, so remove them
  theme_void()
p_tSNE
# It seems like there are 3 or 4 clumps of high-D (4D in our case) data.

# Partitioning Clustering --------------------------------------------------------

# Partitioning clustering tries to classify observations into mutually exclusive
# clusters, such that observations within the same cluster are as similar as
# possible (i.e., high intra-class similarity), whereas observations from
# different clusters are as dissimilar as possible (i.e., low inter-class
# similarity)

# We will be using k-means clustering, which is a popular partitioning
# clustering method.

km_spec <- k_means(
  mode = "partition",
  engine = "stats",
  num_clusters = tune() # we will... tune this?
) |>
  set_args(nstart = 20) # how many random starting centers
# NOTE: If nstart > 1, then K-means clustering will be performed using multiple
# random assignments in Step 1 of the Algorithm, and kmeans() will report only
# the best results.
# It is recommended always running K-means clustering with a large start, such
# as 20 or 50, since otherwise an undesirable local optimum may be obtained.

km_wf <- workflow(preprocessor = rec, spec = km_spec)


## Tuning the number of clusters -----------------
# https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/

# Unlike supervised learning, there is no "ground truth" to compare our clusters
# to. Instead, we can use methods that evaluate the quality of the clustering
# based on the data itself.

km_tuner <- tune_cluster(
  km_wf,
  resamples = vfold_cv(USArrests, v = 5),
  grid = tibble(num_clusters = 1:10),
  metrics = cluster_metric_set(silhouette_avg, sse_within_total)
)

## Elbow method: drop in within-cluster variance:
autoplot(km_tuner, metric = "sse_within_total")
# Note that, the elbow method is sometimes ambiguous (like here).

## Average silhouette method: Measures the quality of a clustering - how well
# each object lies within its cluster. A high average silhouette width indicates
# a good clustering.
autoplot(km_tuner, metric = "silhouette_avg")


# Let's go with 4 clusters, which seems to be a reasonable choice here
km_spec <- km_spec |> finalize_model(list(num_clusters = 4))
km_wf <- km_wf |> update_model(km_spec)
km_fit <- km_wf |> fit(data = USArrests)

## Examining the results  -----------------

extract_centroids(km_fit) # the clusters' centers (on the preprocessed data)
# We can now label the clusters based on these centers.
# ... and extract the assignment of observations to each cluster:
km_clusters <- extract_cluster_assignment(
  km_fit,
  labels = c("High Rural", "High Urban", "Low Urban", "Low Rural")
)

# Number of observations in each cluster
table(km_clusters$.cluster)

# Add the cluster labels to the original data:
USArrests$km_cluster <- km_clusters$.cluster

# We can also find the centers on the original scale:
USArrests |>
  group_by(km_cluster) |>
  summarise(across(Murder:Rape, mean))


# Plotting each observation colored according to its cluster assignment:
plot(USArrests[, 2:5], col = USArrests$km_cluster, pch = 20, cex = 2)

# How does this look on our t-SNE plot?
p_tSNE + aes(color = USArrests$km_cluster)


## More partitioning methods ---------------------------

# The {cluster} package provides several other clustering algorithms (which
# don't work with {tidyclust} just yet), such as PAM and CLARA. See:
# https://cran.r-project.org/web/packages/cluster/refman/cluster.html

# Hierarchical Clustering ---------------------------------------------------------

# Unlike k-means, we don't need to decide on the number of clusters in advance.
# We do however need to decide on:
# 1. A dissimilarity metric
# 2. A linkage method

# There are several dissimilarity metrics to choose from....
philentropy::getDistMethods()
?philentropy::distance()

# We can also use a type of unsupervised random forest to get a distance matrix:
# https://gradientdescending.com/unsupervised-random-forest-example/
dist_rf <- function(x) {
  rf <- randomForest::randomForest(
    x = x,
    mtry = sqrt(ncol(x)),
    ntree = 1000,
    proximity = TRUE
  )
  return(1 - rf$proximity)
}


# We also have serval linkage methods to choose from. See:
?hclust

# We will use the Euclidean distance (which is the default) with complete
# linkage, but you should feel free to experiment with other dissimilarity
# metrics / linkages and see how they affect your results.
hc_spec <- hier_clust(
  mode = "partition",
  engine = "stats",

  # we can actually select this later!
  num_clusters = NULL,
  cut_height = NULL,

  # Decisions!
  linkage_method = "complete",
  dist_fun = partial(distance, method = "euclidean") # (this is the default)
)


hc_fit <-
  workflow(preprocessor = rec, spec = hc_spec) |>
  fit(data = USArrests)

## Exploring the dendrogram ---------------------------
# _Now_ we can plot the dendrogram (with small-ish samples) and decide on the
# number of clusters.

hc_eng <- extract_fit_engine(hc_fit)
plot(hc_eng, labels = USArrests$state.name)
# The numbers / labels at the bottom of the plot identify each observation.

# We can also use:
rect.hclust(hc_eng, k = 4)

plot(hc_eng, labels = USArrests$state.name)
rect.hclust(hc_eng, h = 2)
abline(h = 2, col = "red", lty = 2)

# We can also extract the cluster centers (on the preprocessed data):
extract_centroids(hc_fit, num_clusters = 4) # Looks similar to k-means centers
extract_centroids(km_fit)

# See also:
# ?cluster::bannerplot(hc_eng)

## Cut the tree and examine the results  -----------------

# Determining the cluster labels for each observation is done by cutting the
# dendrogram, either at a specific height (h/cut_height) or by specifying the
# number of clusters (k/num_clusters).
hc_clusters.k4 <- hc_fit |>
  extract_cluster_assignment(
    num_clusters = 4,
    # cut_height = ,
    labels = c("High Rural", "High Urban", "Low Urban", "Low Rural")
  )


# Add the cluster labels to the original data:
USArrests$hc_cluster.k4 <- hc_clusters.k4$.cluster

# Plotting each observation colored according to its cluster assignment:
plot(USArrests[, 2:5], col = USArrests$hc_cluster.k4, pch = 20, cex = 2)

# How does this look on our t-SNE plot?
p_tSNE + aes(color = factor(USArrests$hc_cluster.k4))

table(
  "H-Clust" = USArrests$hc_cluster.k4,
  "K-mean" = USArrests$km_cluster
)

# See also:
?reconcile_clusterings_mapping

# Model based clustering -----------------------------------
# https://mclust-org.github.io/mclust/
?gm_clust
?db_clust


# Validating Clusters ------------------------------------------------------
# All clustering methods will produce **some** clusters, even if there is no
# real or meaningful structure in the data. So, it is important to validate the
# clusters. Read more about this here:
# https://www.fharrell.com/post/cluster/
#
# Unfortunately, {tidyclust} doesn't have built-in functions for cluster
# validation, but we can use other packages for this.

# We will be looking at the k-mean results, but all of these methods can be
# applied to the hierarchical clustering results as well.

## Internal Validation? -------------------------------------------------
# How "good" is the clustering?
# There are several metrics to evaluate clustering quality on the training data.

### Silhouette width ----------------------------
# Silhouette width measures how similar an observation is to its own cluster
# compared to other clusters. The silhouette value ranges from -1 to 1, where a
# value close to 1 indicates that the observation is well clustered, a value
# close to 0 indicates that the observation is on the boundary between two
# clusters, and a value close to -1 indicates that the observation may have been
# assigned to the wrong cluster.

d_euc <- distance(USArrests_z, method = "euclidean") |> as.dist()

sil_km <- cluster::silhouette(
  x = as.integer(USArrests$km_cluster),
  dist = d_euc # k-means uses Euclidean distance
)

plot(sil_km, col = c("purple2", "orange2", "red2", "royalblue2"))
abline(v = c(0.3, 0.5), col = c("red", "blue"), lty = 2)

summary(sil_km)
# Overall the values are adequate (sil>~0.3), but not amazing (ideally we would
# want sil>0.5)...

### Cluster stability ----------------------------
# Are the clusters stable to small perturbations in the data? In other words,
# if we re-sample the data, will we get similar clusters?

library(fpc)

fit_kmeans <- function(X) {
  # This function takes in a data frame X and returns a list the is required by
  # {fpc} functions see e.g., ?kmeansCBI
  .fit <- fit(km_wf, data = X)

  .cl <- extract_cluster_assignment(.fit) |>
    pull(.cluster) |>
    as.integer()

  list(
    results = .fit,
    nc = 4,
    clusterlist = data.frame(outer(.cl, 1:4, "==")),
    partition = rep(1, nrow(X))
  )
}

jac_kmeans <- clusterboot(
  USArrests,
  B = 200,
  datatomatrix = FALSE,
  bootmethod = "boot",

  clustermethod = fit_kmeans
)

# Jaccard values larger than 0.75 indicate stable clusters, while values below
# 0.5 indicate highly unstable clusters.
jac_kmeans # all clusters are stable (low dissolved / unrecovered clusters)
plot(jac_kmeans)

# See also:
?prediction.strength


## External Validation? -------------------------------------------------
# Are these clusters *useful*?
# So far we've seen how the clusters map *back onto* the variables that went
# into the clustering - but these results are trivial. The question is can these
# clusters be used to tell us something new about other variables?

# Let's compare them to some data about the states:
states_info <- read.csv("states_vote1981.csv") |>
  left_join(USArrests, by = join_by(state == state.name))
head(states_info)


# Are the clusters related to voting for Reagan in 1981?
ggplot(states_info, aes(margin_pct, km_cluster, fill = km_cluster)) +
  geom_vline(xintercept = 0) +
  geom_violin() +
  scale_x_continuous("% margin voted for Reagan")
# What can we learn from this?

# Compare cluster means ---------------------------------------------------
# Looking at the cluster centers form k-means:
extract_centroids(km_fit)
# Do the clusters differ on the variables that went into the clustering?

# Do clusters 1 and 2 actually differ on these variables?
# We can't just compute t-tests, because that would be doing post-selection
# inference - the clusters are defined based on the distance on these variables,
# so of course they will differ!
t.test(
  Murder ~ km_cluster,
  data = USArrests,
  subset = km_cluster %in% c("High Rural", "High Urban")
)

# Thankfully, the {clusterpval} package provides functions to test for
# differences between clusters while accounting for the fact that the clusters
# were formed based on the distance on these same variables:

# pak::pak("lucylgao/clusterpval")
library(clusterpval)

# Make a function to obtain clusters:
km_cluster_x <- function(X) {
  # This function takes the *preprocessed* data and returns the cluster
  # assignments as integers.

  # K-means clustering with 4 clusters:
  km_spec |>
    # "fit" the model to the data:
    fit_xy(X) |>
    # Extract the cluster assignment:
    extract_cluster_assignment() |>
    # pull the cluster labels and convert to integer:
    pull(.cluster) |>
    as.integer()
}


test_clusters_approx(
  X = as.matrix(USArrests_z),
  # Clusters to compare
  k1 = 1,
  k2 = 2,

  cl_fun = km_cluster_x,
  cl = as.integer(USArrests$km_cluster),

  ndraws = 200
)

# How about clusters 1 and 4?
test_clusters_approx(
  X = as.matrix(USArrests_z),
  # Clusters to compare
  k1 = 1,
  k2 = 4,

  cl_fun = km_cluster_x,
  cl = as.integer(USArrests$km_cluster),

  ndraws = 200
)


# Exercise ----------------------------------------------------------------

data("oils", package = "modeldata")
?modeldata::oils

# 0. Build a t-SNE plot based on all columns (minus "class").
#   - Color the point by class
# 1. Cluster oils into groups based on these data
#   - use hclust
#   - decide on a distance metric
#   - choose a linkage method
#   - plot the dendrogram - and choose the number of clusters
#   - plot the clusters on a t-SNE plot.
# 2. Validate the clusters:
#   - Are the results stable?
#   - Do the clusters differ on the variables used to create them?
#   - Do the clusters map nicely onto the "class" variable?
