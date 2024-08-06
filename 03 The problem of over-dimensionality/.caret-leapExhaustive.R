# see https://topepo.github.io/caret/using-your-own-model-in-train.html

leapExhaustive <- caret::getModelInfo(model = "leapForward", regex = FALSE)[[1]]

leapExhaustive$label <- "Linear Regression with Exhaustive Selecton"

leapExhaustive$fit <- function (x, y, wts, param, lev, last, classProbs, ...) {
  stopifnot(
    "'nbest' should not be specified" = !"nbest" %in%...names(),
    "'method' should not be specified" = !"method" %in%...names(),
    "'nvmax' should not be specified" = !"nvmax" %in%...names()
  )
  
  if (is.null(wts)) wts <- rep(1, length(y))
  
  leaps::regsubsets(as.matrix(x), y, weights = wts,
                    nbest = 1, nvmax = param$nvmax, 
                    method = "exhaustive", ...)
}

# leapExhaustive$loop #?
leapExhaustive$predict <- function (modelFit, newdata, submodels = NULL, id = NULL) {
  newdata <- as.matrix(newdata)
  newdata <- cbind(rep(1, nrow(newdata)), newdata)
  colnames(newdata)[1] <- "(Intercept)"
  
  betas <- coef(modelFit, id = 1:(modelFit$nvmax - 1))
  
  foo <- function(b, x) x[, names(b), drop = FALSE] %*% b
  out <- foo(betas[[length(betas)]], newdata)[, 1]
  
  if (!is.null(submodels)) {
    numTerms <- unlist(lapply(betas, length))
    if ("(Intercept)" %in% names(betas[[length(betas)]])) 
      numTerms <- numTerms - 1
    
    keepers <- rev(which(numTerms %in% submodels$nvmax))
    if (length(keepers) != length(submodels$nvmax)) 
      stop("Some values of 'nvmax' are not in the model sequence.")
    
    preds <- lapply(betas[keepers], foo, x = newdata)
    preds <- do.call("cbind", preds)
    out <- as.data.frame(cbind(out, preds), stringsAsFactors = TRUE)
    out <- as.list(out)
  } 
  out
}

cat("Use the `leapExhaustive` object in `train(..., method = leapExhaustive)`.")
