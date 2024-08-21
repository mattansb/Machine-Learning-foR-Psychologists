
# https://easystats.github.io/parameters/articles/efa_cfa.html

library(tidyverse)

library(psych)
library(parameters)
library(factoextra) # https://rpkgs.datanovia.com/factoextra/index.html
library(performance)


# Data --------------------------------------------------------------------

# 24 psychological tests given to 145 seventh and eight-grade children
Harman74 <- read.csv("Harman74.csv")
head(Harman74)


# PCA ---------------------------------------------------------------------

PCA_params <- principal_components(Harman74, n = "max", sort = TRUE)
PCA_params[,1:10] # too large to print

# Extract the underlying PCA "model":
PCA_model <- attr(PCA_params, "model")



## How many components to keep? ---------------------
# See also
?parameters::n_components


### Scree plot --------------------------------
scree(Harman74, factors = FALSE, pc = TRUE)
# 2/5 seems to be supported by the elbow method,
# and 5 by the Kaiser criterion.


### Variance accounted for --------------------

get_eigenvalue(PCA_model) |> 
  ggplot(aes(seq_along(cumulative.variance.percent), cumulative.variance.percent)) + 
  geom_point() + 
  geom_line(aes(group = NA)) + 
  geom_hline(yintercept = 90)
# Or 16....



## Extract component scores ----------------------

PCs <- predict(PCA_model, newdata = Harman74) # returns all PCs
head(PCs[,1:2])


## Plots -----------------

fviz_pca_biplot(PCA_model, axes = c(1, 2))






# FA ----------------------------------------------------------------------

# Is the data suitable for FA? 
round(cor(Harman74), 2) # hard to visually "see" structure in the data...

check_sphericity_bartlett(Harman74)



# We will be using pa method with oblimin rotation.


## How many factors? -------------------------
# See also:
?parameters::n_factors


scree(Harman74, factors = TRUE, pc = FALSE)
# 1 / 4 seem to be supported by the elbow
# 2 seem to be supported by the Kaiser criterion.



## Run Factor Analysis (FA) ------------------------------------------------


EFA <- fa(Harman74, nfactors = 4, 
          fm = "pa", # (principal factor solution), or use fm = "minres" (minimum residual method)
          rotate = "oblimin") # or rotate = "varimax"
# You can see a full list of rotation types here:
?GPArotation::rotations


## Understanding the results -----------------------------------------------

EFA 
# Read about the outputs here: https://m-clark.github.io/posts/2020-04-10-psych-explained/
#
# A better output:
model_parameters(EFA, sort = TRUE, threshold = 0.45)
# These give the pattern matrix




fa.diagram(EFA, cut = 0.45) # factor loading plot

# A biplot:
biplot(EFA, choose = c(1,2), pch = ".", cuts = 0.45)
# choose = NULL to look at all of them



## Extract factor scores ----------------------

# We can now use the factor scores just as we would any variable:
data_scores <- predict(EFA, data = Harman74)
colnames(data_scores) <- c("Verbal","Visual","Math","Je Ne Sais Quoi") # name the factors
head(data_scores)






## Reliability -------------------------------------------------------------

# We need a little function here...
efa_reliability <- function(x, keys = NULL, threshold = 0, labels = NULL) {
  #'         x - the result from psych::fa()
  #'      keys - optional, see ?psych::make.keys
  #' threshold - which values from the loadings should be used
  #'    labels - factor labels
  
  L <- unclass(x$loadings)
  r <- x$r  
  
  if (is.null(keys)) keys <- sign(L) * (abs(L) > threshold) 
  
  out <- data.frame(
    Factor = colnames(L),
    Omega = colSums(keys * L)^2 / diag(t(keys) %*% r %*% keys)
  )
  
  if (!is.null(labels))
    out$Factor <- labels
  else
    rownames(out) <- NULL
  
  out
}

efa_reliability(EFA, threshold = 0.45, 
                labels = c("Verbal","Visual","Math","Je Ne Sais Quoi"))
# These are interpretable similarly to Cronbach's alpha


