order_rows_by_fold <- function(Row, Fold, Data) {
  Fold.id <- split(data.frame(Data, Fold), Row) |>
    lapply(function(x) {
      x$Fold[x$Data == "Assessment"]
    }) |>
    unsplit(Row)
  Fold.id <- regmatches(Fold.id, m = regexpr("[0-9]+", Fold.id))
  reorder(factor(Row), as.integer(Fold.id))
}
