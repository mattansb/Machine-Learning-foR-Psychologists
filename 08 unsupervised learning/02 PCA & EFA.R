library(psych)

library(parameters)
library(performance)

# recall the difference between clustering and PCA/EFA:
# - clustering looks to find homogeneous subgroups among the observations.
# - PCA and EFA looks to find a low-dimensional representation of the obs. that
#   accounts for a good fraction of their variance.

# Data --------------------------------------------------------------------

# 24 psychological tests given to 145 seventh and eight-grade children
Harman74 <- read.csv("Harman74.csv")
head(Harman74)

# Based on:
?datasets::Harman74.cor


# PCA ---------------------------------------------------------------------

## How many components (to keep)? ---------------------
# See also
?parameters::n_components


# Scree plot:
sp <- scree(Harman74, factors = FALSE, pc = TRUE)
# 2/5 seems to be supported by the elbow method,
# and 5 by the Kaiser criterion.

# Variance accounted for:
pcv <- sp$pcv
pcv_pct <- pcv / sum(pcv) # normalize


plot(
  seq_along(pcv_pct),
  cumsum(pcv_pct),
  xlab = "PC",
  ylab = "Cumulative variance accounted for"
)
axis(side = 1, at = seq_along(pcv_pct))
abline(a = 0.9, b = 0)
# Or 16....

## PCA decomposition ----------------------

pca5_fit <- principal(Harman74, nfactors = 5, rotate = "none")


## Understanding the components -------------------

model_parameters(pca5_fit)
# Note the loadings are highest on the first factor, and so forth...

biplot(pca5_fit, choose = 2)
biplot(pca5_fit, choose = c(1, 2))


## Extract components scores ------------------

predict(pca5_fit, data = Harman74[1:5, ]) # n-by-p matrix


# (E)FA ----------------------------------------------------------------------

# https://easystats.github.io/parameters/articles/efa_cfa.html

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

fa.parallel(Harman74, fa = "fa", error.bars = TRUE, sim = FALSE, fm = "pa")
axis(side = 1, at = 1:15)
abline(a = 0.5, b = 0, col = "green4")


## (Exploratory) factor analysis ------------------------------------------------

efa4_fit <- fa(
  Harman74,
  nfactors = 4,
  fm = "pa", # (principal factor solution), or use fm = "minres" (minimum residual method)
  rotate = "oblimin"
) # or rotate = "varimax"
# You can see a full list of rotation types here:
?GPArotation::rotations


## Understanding the factors -----------------------------------------------

efa4_fit
# Read about the outputs here: https://m-clark.github.io/posts/2020-04-10-psych-explained/
# Fit measures:
efa4_fit[c("RMSEA", "TLI")]

# A better output:
model_parameters(efa4_fit, sort = TRUE, threshold = 0.45)
# These give the pattern matrix

fa.diagram(efa4_fit, cut = 0.45) # factor loading plot

# A biplot:
biplot(efa4_fit, choose = c(1, 2), pch = ".", cuts = 0.45)
# choose = NULL to look at all of them

## Extract factor scores ----------------------

# We can now use the factor scores just as we would any variable:
data_scores <- predict(efa4_fit, data = Harman74) |> data.frame()
colnames(data_scores) <- c("Verbal", "Visual", "Math", "Je Ne Sais Quoi") # name the factors
head(data_scores)


## Reliability -------------------------------------------------------------

# We need a little function here...
efa_reliability <- function(x, keys = NULL, threshold = 0, labels = NULL) {
  #' @param x The result from [psych::fa()]
  #' @param keys Optional, see ?[psych::make.keys]
  #' @param threshold When `keys = NULL`; which values from the loadings should
  #'   be used
  #' @param labels Factor labels

  L <- unclass(x$loadings)
  r <- x$r

  if (is.null(keys)) {
    keys <- sign(L) * (abs(L) > threshold)
  }

  out <- data.frame(
    Factor = labels %||% colnames(L),
    Omega = colSums(keys * L)^2 / diag(t(keys) %*% r %*% keys)
  )

  rownames(out) <- NULL

  out
}


efa_reliability(
  efa4_fit,
  threshold = 0.45,
  labels = c("Verbal", "Visual", "Math", "Je Ne Sais Quoi")
)
# These are interpretable similarly to Cronbach's alpha

# Exercise ----------------------------------------------------------------

library(dplyr)
library(tidyr)

# Select only the 25 first columns corresponding to the items on the BIG-5
# scales:
data("bfi", package = "psychTools")
?psychTools::bfi

bfi_tidy <- bfi |>
  mutate(
    gender = factor(gender, labels = c("Male", "Female")),
    education = factor(
      education,
      labels = c(
        "HS",
        "finished HS",
        "some college",
        "college graduate",
        "graduate degree"
      )
    )
  ) |>
  drop_na(1:25)

head(bfi_tidy)

bfi_scales <- bfi_tidy |>
  select(1:25)

## A. PCA
# What is the minimal number of components that can be used to represent 85% of
# the variance in the bfi scale?

## B. EFA:
# 1. Validate the big-5: look at a scree-plot to see if the data suggests 5
#   factors or more or less.
# 2. Conduct an EFA
# 3. Look at the loadings - do they make sense?
