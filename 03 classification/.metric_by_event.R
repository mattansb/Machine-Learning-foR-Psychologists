metric_by_event <- function(
  data,
  metric_set,
  truth,
  estimate,
  ...,
  .all = FALSE
) {
  # Only for class_prob_metric_set
  stopifnot(inherits(
    metric_set,
    c("class_prob_metric_set", "prob_metric", "class_metric")
  ))

  # Make a call with the metric set
  cl <- match.call(expand.dots = FALSE)
  cl[[1]] <- as.name(as.character(cl$metric_set))
  cl[["..."]] <- NULL
  cl[["metric_set"]] <- NULL
  cl[["event_level"]] <- "first"

  # Find names and levels
  truth_name <- as.character(cl$truth)
  levels <- levels(data[[truth_name]])

  estimate_name <- as.character(cl$estimate)
  prob_names <- as.character(substitute(list(...)))[-1]
  has_est <- length(estimate_name) > 0L
  has_prob <- all(nzchar(prob_names))

  # A function to binarize a multiclass factor
  fct_binarize <- function(.f, lvl) {
    forcats::fct_collapse(.f, Yes = lvl, other_level = "No")
  }

  # For each class...
  out <- levels |>
    purrr::imap(function(lvl, i) {
      tmp_data <- data
      tmp_cl <- cl

      # binarize
      tmp_data[[truth_name]] <- fct_binarize(tmp_data[[truth_name]], lvl)
      if (has_est) {
        tmp_data[[estimate_name]] <- fct_binarize(
          tmp_data[[estimate_name]],
          lvl
        )
      }
      if (has_prob) {
        tmp_cl[[length(tmp_cl) + 1]] <- as.name(prob_names[i])
      }
      tmp_cl$data <- tmp_data

      # Get the metrics
      eval.parent(tmp_cl)
    }) |>
    purrr::set_names(nm = levels) |>
    # rbind with an `.class` column for each class
    dplyr::bind_rows(.id = ".class") |>
    dplyr::mutate(
      # New estimator name?
      .estimator = "event"
    )

  if (isFALSE(.all)) {
    # Get current default behavior
    if (has_prob) {
      cl[length(cl) + seq_along(prob_names)] <- lapply(prob_names, as.name)
    }
    out_raw <- eval.parent(cl) |>
      dplyr::filter_out(
        .estimator %in% c("macro", "macro_weighted", "micro", "hand_till")
      )

    # keep only metrics that use macro/micro multiclass estimators
    out <- out |>
      dplyr::anti_join(out_raw, by = dplyr::join_by(.metric)) |>
      # Add the (non-classwise) metrics
      bind_rows(out_raw)
  }

  # Deal with grouped data frames
  if (inherits(data, "grouped_df")) {
    out <- out |>
      dplyr::relocate(dplyr::all_of(dplyr::group_vars(data)), .before = 1) |>
      dplyr::arrange(dplyr::pick(dplyr::all_of(dplyr::group_vars(data))))
  }

  out
}
