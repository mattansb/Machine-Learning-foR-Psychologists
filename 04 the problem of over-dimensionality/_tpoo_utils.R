predict.regsubsets <- function(
  object,
  newdata,
  id = NULL,
  select = c("adjr2", "cp", "bic")
) {
  cl <- object$call
  cl[[1]] <- quote(stats::lm)
  cl[!names(cl) %in% c(formalArgs(stats::lm), "")] <- NULL
  lm_object <- eval.parent(cl)

  X_newdata <- model.matrix(
    terms(lm_object),
    newdata,
    contrasts.arg = object$contrasts
  )

  if (is.null(id)) {
    select <- match.arg(select)
    v <- summary(object)[[select]]
    id <- switch(select, adjr2 = which.max(v), cp = , bic = which.min(v))
  }

  b <- coef(object, id = id)

  as.vector(X_newdata[, names(b), drop = FALSE] %*% b)
}

plot_glmnet_coef <- function(mod, s = 0, show_intercept = FALSE, ...) {
  b <- glmnet::coef.glmnet(mod, s = sort(unique(c(0, s))), ...) |>
    as.matrix() |>
    as.data.frame() |>
    tibble::rownames_to_column("Coef")

  if (isFALSE(show_intercept)) {
    b <- b |> filter(Coef != "(Intercept)")
  }

  ggplot2::ggplot(b, ggplot2::aes(Coef, .data[[tail(colnames(b), 1)]])) +
    ggplot2::geom_blank(ggplot2::aes(y = .data[[colnames(b)[2]]])) +
    ggplot2::geom_hline(yintercept = 0) +
    ggplot2::geom_segment(
      ggplot2::aes(
        xend = Coef,
        yend = .data[[colnames(b)[2]]]
      ),
      color = "grey50"
    ) +
    ggplot2::geom_point(
      ggplot2::aes(shape = .data[[tail(colnames(b), 1)]] == 0),
      fill = "red",
      size = 2,
      show.legend = c(shape = TRUE)
    ) +
    ggplot2::scale_shape_manual(
      NULL,
      breaks = c(FALSE, TRUE),
      values = c(16, 24),
      labels = c("none-0", "0"),
      limits = c(FALSE, TRUE)
    ) +
    ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(angle = 30)) +
    ggplot2::labs(y = "Coef", x = NULL) +
    ggplot2::ggtitle(bquote(lambda == .(s)))
}

extract_pcr_coef <- function(x, which = c("coef", "vcov")) {
  stopifnot(
    "x is not a workflow" = inherits(x, "workflow"),
    "the model spec is not linear_reg / logistic_reg" = inherits(
      x_spec <- hardhat::extract_spec_parsnip(x),
      c("linear_reg", "logistic_reg")
    ),
    "last step in recipe is not step_pca" = inherits(
      x_step_pca <- tail(hardhat::extract_recipe(x)$steps, 1)[[1]],
      "step_pca"
    )
  )

  which <- match.arg(which)

  x_prcomp <- x_step_pca$res
  rotation <- x_prcomp$rotation
  # we only want the first k of these:
  loadings_pcr <- rotation[, 1:x_step_pca$num_comp]

  # get the coefficients and standard errors on the PC scales:
  pcr_eng <- hardhat::extract_fit_engine(x)
  b <- coef(pcr_eng)
  V <- vcov(pcr_eng)

  if (which == "coef") {
    # These are the coefficients on the original scale
    drop(b[-1] %*% t(loadings_pcr))
  } else {
    # These are the standard errors
    loadings_pcr %*% V[-1, -1] %*% t(loadings_pcr)
  }
}


extract_pls_coef <- function(x, which = c("coef", "vcov")) {
  stopifnot(
    "x is not a workflow" = inherits(x, "workflow"),
    "the model spec is not linear_reg / logistic_reg" = inherits(
      hardhat::extract_spec_parsnip(x),
      c("linear_reg", "logistic_reg")
    ),
    "last step in recipe is not step_pls()" = inherits(
      x_step_pls <- tail(hardhat::extract_recipe(x)$steps, 1)[[1]],
      "step_pls"
    )
  )

  which <- match.arg(which)

  res <- x_step_pls$res
  sd_x <- res$sd
  loadings_pls <- mapply("*", as.data.frame(res$coefs), res$col_norms)
  # we only want the first k of these:
  loadings_pls <- loadings_pls[, 1:x_step_pls$num_comp, drop = FALSE]

  # get the coefficients and standard errors on the PC scales:
  pls_eng <- hardhat::extract_fit_engine(x)
  b <- coef(pls_eng)
  V <- vcov(pls_eng)

  if (which == "coef") {
    # These are the coefficients on the original scale
    drop(b[-1] %*% t(loadings_pls)) / sd_x
  } else {
    # These are the standard errors
    ((loadings_pls %*% V[-1, -1] %*% t(loadings_pls))) * (sd_x %*% t(sd_x))
  }
}
