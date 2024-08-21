

#' @param truth A `factor` vector
#' @param estimate A numeric vector of class probabilities corresponding to the
#'   "relevant" class, or a 2-column matrix (it is assumed that the first column
#'   corresponds to the first level of `truth`).
#' @param threshold A threshold (0-1) used to factorize the probabilities of the
#'   `event_level` event.
#' @param class_metric One of the `*_vec()` class metric functions.
#' @param ... Arguments passed to `class_metric`.
#' @param estimator Must be "binary".
#' @param event_level A single string. Either "first" or "second" to specify
#'   which level of truth to consider as the "event".
rethresh_vec <- function(truth,
                         estimate,
                         threshold = 0.5,
                         class_metric = yardstick::accuracy_vec,
                         ...,
                         estimator = "binary",
                         event_level = "first") {
  stopifnot(estimator == "binary", 0 < threshold && threshold < 1)
  
  # prob estimate
  if (is.matrix(estimate) && ncol(estimate) > 2L) {
    stop("rethresh() only supports binary classifier!")
  } else if (is.matrix(estimate)  && ncol(estimate) == 2L) {
    if (event_level == "second") {
      estimate <- estimate[, 2]
    } else {
      estimate <- estimate[, 1]
    }
  }
  estimate <- as.vector(estimate)
  
  if (event_level == "second") {
    estimate <- 1 - estimate
  } else {
    threshold <- 1 - threshold
  }
  
  estimate <- probably::make_two_class_pred(estimate,
                                            threshold = 1 - threshold,
                                            levels = levels(truth))
  
  do.call(what = class_metric, args = c(
    list(
      truth = truth,
      estimate = estimate,
      event_level = event_level
    ),
    list(...)
  ))
}

#' @param data A `data.frame` containing the columns specified by `truth` and
#'   `...`.
#' @param truth The column identifier for the true class results (that is a
#'   `factor`).
#' @param ... One or two unquoted column names (or one or more dplyr selector
#'   functions) to choose which variables in `data` contain the class
#'   probabilities.
#' @param threshold A threshold (0-1) used to factorize the probabilities of the
#'   `event_level` event.
#' @param class_metric One of the `*_vec()` class metric functions.
#' @param options A list of named arguments passed to `class_metric`.
#' @param event_level A single string. Either "first" or "second" to specify
#'   which level of truth to consider as the "event" for thresholding. Also
#'   passed to `class_metric`.
#' @param estimator Must be "binary".
#' @param na_rm,case_weights Arguments passed to `class_metric`.
rethresh <- function(data,
                     truth,
                     ...,
                     threshold = 0.5,
                     class_metric = yardstick::accuracy_vec,
                     options = list(),
                     estimator = "binary",
                     na_rm = TRUE,
                     case_weights = NULL,
                     event_level = "first") {
  yardstick::prob_metric_summarizer(
    name = "rethresh",
    fn = rethresh_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    ...,
    estimator = estimator,
    na_rm = na_rm,
    event_level = event_level,
    case_weights = !!rlang::enquo(case_weights),
    fn_options = c(
      options,
      list(class_metric = class_metric, threshold = threshold)
    )
  )
}

# a version the should be used when `class_metric` should be minimized
rethresh_minimizer <- yardstick::new_prob_metric(rethresh, "minimize")
# a version the should be used when `class_metric` should be maximized
rethresh <- yardstick::new_prob_metric(rethresh, "maximize")




message(paste(
  "---",
  "Use rethresh_vec() or rethresh() (or rethresh_minimizer() with a minimizing metric).",
  "",
  "e.g.,:",
  '  accuracy0.4 <- metric_tweak("accuracy0.4", .fn = rethresh,',
  "                              threshold = 0.4,",
  "                              class_metric = accuracy_vec)",
  "",
  "See file for more examples.", 
  "---",
  
  sep = "\n"
))


#' # Examples ----------------------------------------------------------------
#' 
#' ## Fit a logistic regression model with tidymodels -----------------
#' 
#' data("mtcars")
#' mtcars$am <- factor(mtcars$am)
#' 
#' logreg_fit <- workflows::workflow(
#'   preprocessor = recipes::recipe(am ~ hp, data = mtcars),
#'   spec = parsnip::logistic_reg(mode = "classification", engine = "glm")
#' ) |>
#'   parsnip::fit(data = mtcars)
#' 
#' 
#' 
#' ## Extract predictions ------------------------------
#' 
#' mtcars_predictions <- parsnip::augment(logreg_fit, new_data = mtcars) |>
#'   dplyr::mutate(
#'     # Add class prediction for a threshold of 0.4
#'     .pred_class0.4 = factor(.pred_1 > 0.4, levels = c(F, T), labels = c("0", "1"))
#'   )
#' 
#' 
#' 
#' ## Evaluate ------------------------------
#' 
#' ### Accuracy @ threshold of 0.4 ---------------
#' mtcars_predictions |> yardstick::accuracy(am, .pred_class0.4)
#' 
#' accuracy0.4 <- yardstick::metric_tweak("acc0.4", .fn = rethresh,
#'                                        threshold = 0.4, class_metric = yardstick::accuracy_vec,
#'                                        event_level = "second")
#' mtcars_predictions |> accuracy0.4(am, .pred_1)
#' mtcars_predictions |> accuracy0.4(am, .pred_0, .pred_1)
#' 
#' 
#' ### f2 measure @ threshold of 0.4 ----------------------------
#' mtcars_predictions |> yardstick::f_meas(am, .pred_class0.4, beta = 2, event_level = "second")
#' f2_meas0.4 <- yardstick::metric_tweak("acc0.4", .fn = rethresh,
#'                                       threshold = 0.4,
#'                                       class_metric = yardstick::f_meas_vec, options = list(beta = 2),
#'                                       event_level = "second")
#' mtcars_predictions |> f2_meas0.4(am, .pred_1)
#' mtcars_predictions |> f2_meas0.4(am, .pred_0, .pred_1)
#' 
