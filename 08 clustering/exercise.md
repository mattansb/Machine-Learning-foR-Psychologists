
# Exercise

Complete the following in R or Python.

## Clustering

```R
data("oils", package = "modeldata")
?modeldata::oils
```

```Python
wine = load_wine(as_frame=True)
wine_data = wine.data
wine_data["target"] = wine.target.astype("category")
print(wine_data.info())
```

1. Build a t-SNE plot based on all columns (minus the true class).
  - Color the point by the true class
2. Cluster observations into groups based on these data
  - use hclust
  - decide on a distance metric
  - choose a linkage method
  - plot the dendrogram - and choose the number of clusters
  - plot the clusters on a t-SNE plot.
3. Validate the clusters:
  - Are the results stable?
  - Do the clusters map nicely onto the "true class" variable?
  - Do the clusters differ on the variables used to create them?
