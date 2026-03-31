### Tutorial 6- Clustering  ###

library(tidymodels)

library(patchwork)

library(cluster)
library(Rtsne)
library(factoextra) # https://rpkgs.datanovia.com/factoextra/index.html

# Learn more at:
# https://tidyclust.tidymodels.org/index.html

# The Data ---------------------------------------

data(USArrests)
?USArrests # Violent Crime Rates by US State

# Are there clusters of states that are similar in their rates of crime?

## Prep -------------------

summary(USArrests)
# variables are on very different scales.
#
# We should re-scale them.

# We can use recipe:
rec <- recipe(~., data = USArrests) |>
  step_normalize(all_numeric_predictors())

USArrests_z <- bake(prep(rec), new_data = USArrests, composition = "data.frame")
rownames(USArrests_z) <- rownames(USArrests) # we need to re-add rownames :/

head(USArrests_z)

plot(USArrests_z, pch = 20, cex = 2)
# there are some associations between variables, but are there CLUSTERS?

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

USArrests_z_tSNE <- Rtsne(USArrests_z, perplexity = 5)
# Default perplexity is 30, but this value is too large for our small dataset.

p_tSNE <- data.frame(USArrests_z_tSNE$Y) |>
  ggplot(aes(X1, X2)) +
  geom_point(size = 2) +
  ggrepel::geom_text_repel(aes(label = rownames(USArrests_z))) +
  # scales are meaningless, so remove them
  theme_void()
p_tSNE
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

## Elbow method: drop in within-cluster variance:
fviz_nbclust(
  USArrests_z,
  FUNcluster = kmeans, # we will use kmeans for clustering
  method = "wss",
  k.max = 20
) +
  geom_vline(xintercept = 4, linetype = 2) +
  labs(subtitle = "Elbow method")
# Note that, the elbow method is sometimes ambiguous (like here).

## Average silhouette method: Measures the quality of a clustering - how well
# each object lies within its cluster. A high average silhouette width indicates
# a good clustering.
fviz_nbclust(
  USArrests_z,
  FUNcluster = kmeans,
  method = "silhouette",
  k.max = 20
) +
  labs(subtitle = "Silhouette method")


## Gap statistic (using bootstrap sampling): measures how far is the observed
# within intra-cluster variation is from a random uniform distribution of the
# data?
fviz_nbclust(
  USArrests_z,
  FUNcluster = kmeans,
  method = "gap_stat",
  k.max = 20
) +
  labs(subtitle = "Gap statistic method")
# Suggests k=2 or k=4

## Finding Clusters with k-means ------------------------------

# The function kmeans() performs K-means clustering.
km <- kmeans(
  USArrests_z, # Features to guide clustering
  centers = 4, # K
  nstart = 20 # how many random starting centers
)
# NOTE: If nstart > 1, then K-means clustering will be performed using multiple
# random assignments in Step 1 of the Algorithm, and kmeans() will report only
# the best results.
# It is recommended always running K-means clustering with a large start, such
# as 20 or 50, since otherwise an undesirable local optimum may be obtained

km$cluster # vector which assigns obs. to each cluster
km$size # obs. num in each cluster

km$centers # the two clusters' centers
# We can also find the centers on the original scale:
USArrests |>
  group_by(km = km$cluster) |>
  summarise(across(everything(), mean))
# Can now label the clusters based on these centers:
cluster_km <- factor(
  km$cluster,
  labels = c("High Rural", "Low Rural", "High Urban", "Low Urban")
)


# Plotting each observation colored according to its cluster assignment:
plot(USArrests, col = cluster_km, pch = 20, cex = 2)

# How does this look on our t-SNE plot?
p_tSNE + aes(color = cluster_km)


# Let's save these results for later:
dta_clust <- tibble(state.name, state.abb, kmeans = cluster_km)


## More partitioning methods ---------------------------

# The {cluster} package provides several other clustering algorithms, such as
# PAM and CLARA. See:
# https://cran.r-project.org/web/packages/cluster/refman/cluster.html

# Hierarchical Clustering ---------------------------------------------------------

# This is a "Bottom Up" approach- cluster features on the basis of the
# observations. NO prior selection of num. of clusters.

# We will use the same data to plot the hierarchical clustering dendrogram using
# complete, single, and average linkage clustering, with Euclidean distance as
# the dissimilarity measure.

## Distance metric -----------------------------------
# There are several to choose from....
?get_dist

USArrests_d_euc <- get_dist(USArrests_z, method = "euclidean")
USArrests_d_cor <- get_dist(USArrests_z, method = "pearson")

# We can also use a type of unsupervised random forest to get a distance matrix:
# https://gradientdescending.com/unsupervised-random-forest-example/
rf <- randomForest::randomForest(
  x = USArrests,
  mtry = sqrt(ncol(USArrests)),
  ntree = 1000,
  proximity = TRUE
)
USArrests_d_rf <- as.dist(1 - rf$proximity)

# Which should we use?
fviz_dist(USArrests_d_euc)
fviz_dist(USArrests_d_cor)
fviz_dist(USArrests_d_rf)


## Build dendrogram -----------------
# The hclust() function implements hierarchical clustering.

hc.complete <- hclust(USArrests_d_euc, method = "complete")
# Or method = "average"
# Or method = "single"
# Or method = "centroid"

# plotting the dendrograms (with small-ish samples) and determining clusters:
plot(hc.complete)
# The numbers / labels at the bottom of the plot identify each observation.

# We can also use:
fviz_dend(hc.complete)
fviz_dend(hc.complete, k = 4)
fviz_dend(hc.complete, h = 3) +
  geom_hline(yintercept = 3, linetype = 2)

# See also:
# bannerplot(hc.complete)

## Cut the tree! -----------------------------------

# To determine the cluster labels for each observation associated with a given
# cut of the dendrogram, we can use the cutree() function:
(hc_cut.k4 <- cutree(hc.complete, k = 4))
(hc_cut.h1 <- cutree(hc.complete, h = 1))


# Plotting each observation colored according to its cluster assignment:
plot(USArrests, col = hc_cut.k4, pch = 20, cex = 2)

# How does this look on our t-SNE plot?
p_tSNE + aes(color = factor(hc_cut.k4))

# Save the results:
dta_clust$hclust_k4 <- factor(hc_cut.k4[state.name])


# Do the methods agree?
USArrests_z |>
  group_by(km = hc_cut.k4) |>
  summarise(across(everything(), mean))
# This looks very similar to k-means results...

table(
  "H-Clust" = dta_clust$hclust_k4,
  "K-mean" = dta_clust$kmeans
)

# Model based clustering -----------------------------------
#
# See the {mclust} package
# https://mclust-org.github.io/mclust/

# Understanding Clusters ------------------------------------------------------
# All clustering methods will produce clusters, even if there is no real
# structure in the data. So, it is important to validate the clusters.

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

sil_km <- silhouette(
  km$cluster,
  dist = USArrests_d_euc # k-means uses euclidean distance
)
plot(sil_km, col = sample(colors(TRUE), 4))
summary(sil_km)
# Overall the values are adequate, but not amazing...

### Cluster stability ----------------------------
# Are the clusters stable to small perturbations in the data? In other words,
# if we re-sample the data, will we get similar clusters?

library(fpc)

jac_kmeans <- clusterboot(
  USArrests_z,
  B = 200,

  clustermethod = kmeansCBI,
  k = 4
)

jac_hclust <- clusterboot(
  USArrests_d_euc,
  B = 200,

  clustermethod = hclustCBI,
  k = 4,
  method = "complete"
)

# Jaccard values larger than 0.75 indicate stable clusters, while values below
# 0.5 indicate highly unstable clusters.

jac_kmeans # all clusters are stable
jac_hclust # cluster 2 and (4?) are unstable


### Compare cluster means ---------------
# Do the clusters differ on the variables that went into the clustering?

# Let's use k-means clusters for this example. Recall:
km$centers
# Do clusters 1 and 3 differ on these variables?
# We cant just compute t-tests, because that would be doing post-selection
# inference!
# Thankfully, the {clusterpval} package provides functions to test for
# differences between clusters while accounting for the fact that the clusters
# were formed based on the distance on these same variables:

# pak::pak("lucylgao/clusterpval")
library(clusterpval)

# Make a function to obtain clusters:
cl_fun = \(X) kmeans(X, centers = 4, nstart = 20)$cluster

test_clusters_approx(
  X = as.matrix(USArrests_z),
  # Clusters to compare
  k1 = 1,
  k2 = 3,

  cl_fun = cl_fun,
  cl = km$cluster
)

# How about clusters 1 and 4?
test_clusters_approx(
  X = as.matrix(USArrests_z),
  # Clusters to compare
  k1 = 1,
  k2 = 4,

  cl_fun = cl_fun,
  cl = km$cluster
)


## External Validation? -------------------------------------------------
# Are these clusters *useful*?
# So far we've seen how the clusters map *back onto* the variables that went
# into the clustering - but these results are trivial. The question is can these
# clusters be used to tell us something new about other variables?

# Let's compare them to some data about the states:
states_info <- read.csv("states_info.csv") |>
  left_join(dta_clust, by = c("state.abb", "state.name"))
head(states_info)


# Are the clusters related to voting Trump in 2024?
p_trump <-
  ggplot(states_info, aes(trump2024, kmeans, fill = kmeans)) +
  geom_vline(xintercept = 0.5) +
  geom_violin() +
  scale_x_continuous("% voted for Trump", limits = c(0, 1))


# Are the clusters related to number of casinos?
p_casino <-
  ggplot(states_info, aes(n_casinos, kmeans, fill = kmeans)) +
  geom_violin()

p_trump + p_casino + plot_layout(guides = "collect")


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
