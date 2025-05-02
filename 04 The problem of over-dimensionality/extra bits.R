
# 03 Dimension reduction / PCR / The final model ----------------------------

# Here is an alternative to using pls::pcr()

extract_step_pca_weights <- function(x, id) {
  # this function finds the PCA rotation weights trained in step_pca()
  
  step_ids <- vapply(x$steps, function(x) x$id, character(1))
  number <- which(id == step_ids)
  .prcomp <- x$steps[[number]]$res
  .prcomp$rotation
}

loadings <- 
  extract_recipe(pcr_fit) |> 
  extract_step_pca_weights(id = "pp-PCA")

# we only want the first k of these:
loadings_pcr <- loadings[,1:best_pcr$num_comp]

# get the coefficients and standard errors on the PC scales:
pcr_eng <- extract_fit_engine(pcr_fit)
b <- coef(pcr_eng)
V <- vcov(pcr_eng)

# These are the coefficients on the original scale
drop(b[-1] %*% t(loadings_pcr))

# These are the standard errors
loadings_pcr %*% V[-1, -1] %*% t(loadings_pcr)

