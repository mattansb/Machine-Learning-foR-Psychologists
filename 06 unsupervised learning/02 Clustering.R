### Tutorial 6- Clustering  ###

library(tidyverse)
library(factoextra) # https://rpkgs.datanovia.com/factoextra/index.html
library(recipes)

# recall the difference between clustering and PCA:
# - PCA looks to find a low-dimensional representation of the obs. that accounts
#   for a good fraction of their variance.
# - clustering looks to find homogeneous subgroups among the observations.



# The Data ---------------------------------------


data(USArrests)
?USArrests # Violent Crime Rates by US State

# Are there clusters of states that are similar in their rates of crime?


## Prep -------------------

datawizard::describe_distribution(USArrests) 
# variables are on very different scales.
#
# We should re-scale them.

USArrests_z <- datawizard::standardize(USArrests)

# We can also use recipe:
rec <- recipe( ~ ., data = USArrests) |>
  step_normalize(all_numeric_predictors())

USArrests_z <- prep(rec) |> bake(new_data = USArrests, 
                                 composition = "data.frame")
rownames(USArrests_z) <- rownames(USArrests)

head(USArrests_z)




# K-Means Clustering --------------------------------------------------------

# k-means clustering tries to classify observations into mutually exclusive
# clusters, such that observations within the same cluster are as similar as
# possible (i.e., high intra-class similarity), whereas observations from
# different clusters are as dissimilar as possible (i.e., low inter-class
# similarity)

# This is a "Top Down" approach - cluster observations on the basis of the
# features. Prior selection of K - number of clusters.



## Finding Clusters ------------------------------


# The function kmeans() performs K-means clustering. 
km <- kmeans(USArrests_z, # Features to guide clustering
             centers = 2, # K
             nstart = 20) # how many random starting centers
# NOTE: If nstart > 1, then K-means clustering will be performed using multiple
# random assignments in Step 1 of the Algorithm, and kmeans() will report only
# the best results.
# It is recommended always running K-means clustering with a large start, such
# as 20 or 50, since otherwise an undesirable local optimum may be obtained


km$centers # the two clusters' centers
km$size # obs. num in each cluster 

km$cluster # vector which assigns obs. to each cluster


# Plotting each observation colored according to its cluster assignment:
plot(USArrests, col = km$cluster, 
     pch = 20, cex = 2)

# Maybe K=3 will be better?
# If we don't have any prior knowledge about k... we can use fviz_nbclust() from
# factoextra lib.





## Choosing number of clusters ---------------------


## Elbow method: drop in within-cluster variance:
fviz_nbclust(
  USArrests, FUNcluster = kmeans, 
  method = "wss", k.max = 20
) +
  geom_vline(xintercept = 3, linetype = 2) +
  labs(subtitle = "Elbow method")
# Note that, the elbow method is sometimes ambiguous.






## Average silhouette method: Measures the quality of a clustering - how well
# each object lies within its cluster. A high average silhouette width indicates
# a good clustering.
fviz_nbclust(
  USArrests, FUNcluster = kmeans, 
  method = "silhouette", k.max = 20
) +
  labs(subtitle = "Silhouette method")


## Gap statistic (using bootstrap sampling)
# Read about it here:
# https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/









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

# Which should we use?

fviz_dist()



## Build dendrogram -----------------
# The hclust() function implements hierarchical clustering.

hc.complete <- hclust(, method = "complete")
# Or method = "average"
# Or method = "single"
# Or method = "centroid"


# plotting the dendrograms and determining clusters:
plot(hc.complete)
# The numbers / labels at the bottom of the plot identify each observation.

# We can also use:
fviz_dend(hc.complete) 

fviz_dend(hc.complete, h = 1) +
  geom_hline(yintercept = 1)

fviz_dend(hc.complete, k = 4)


## Cut the tree! -----------------------------------

# To determine the cluster labels for each observation associated with a given
# cut of the dendrogram, we can use the cutree() function:
cutree(hc.complete, k = 4)
cutree(hc.complete, h = 1)






# Exercise ----------------------------------------------------------------

# Select only the 25 first columns corresponding to the items on the BIG-5
# scales:
data("bfi", package = "psychTools")

bfi <- bfi |> 
  mutate(
    gender = factor(gender, labels = c("Male", "Female")),
    education = factor(education, labels = c("HS", "finished HS", "some college", "college graduate", "graduate degree"))
  ) |>
  drop_na(1:25)

head(bfi)

bfi_scales <- bfi |> 
  select(1:25)


## A. Clustering
# 1. Cluster people into groups based on these data
#   - use hclust
#   - decide on a distance metric
#   - choose a linkage method
#   - plot the dendrogram - and choose the number of clusters
# 2. Validate the clusters - are they related to gender? age? education?
#   - Answer visually.

## B. PCA
# What is the minimal number of components that can be used to represent 85% of
# the variance in the bfi scale?

## C. EFA (bonus)
# 1. Validate the big-5: look at a scree-plot to see if the data suggests 5
#   factors or more or less.
# 2. Conduct an EFA.
# 3. Look at the loadings - do they make sense?
