# From:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html


def se_bound(cv_results, n_fold, greater_is_better=True, scorer=None):
    """
    Calculate the lower/upper bound within 1 standard deviation
    of the best `mean_test_{scorer}`.

    Parameters
    ----------
    cv_results : dict
        Dictionary of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`
    n_fold : int
        Number of folds used
    greater_is_better : bool
        Is more better?
    scorer : str, optional
        Name of scorer

    Returns
    -------
    float
        Lower/upper bound within 1 standard deviation of the
        best `mean_test_{scorer}`.
    """

    import pandas as pd
    import numpy as np

    cv_results = pd.DataFrame(cv_results)

    if scorer is None:
        scorer = "score"

    if greater_is_better:
        best_score_idx = np.argmax(cv_results[f"mean_test_{scorer}"])
    else:
        best_score_idx = np.argmin(cv_results[f"mean_test_{scorer}"])

    M = cv_results[f"mean_test_{scorer}"][best_score_idx]
    S = cv_results[f"std_test_{scorer}"][best_score_idx] / np.sqrt(n_fold)

    if greater_is_better:
        return M - S
    else:
        return M + S


def make_one_se_rule_selector(
    n_fold,
    by,
    ascending,
    greater_is_better=True,
    scorer=None,
    ret_index=True,
):
    """
    Balance model complexity with cross-validated score.

    Parameters
    ----------

    by : str list
        see pd.DataFrame.sort_values()
    ascending : bool list
        see pd.DataFrame.sort_values()
    n_fold : int
        Number of folds used
    greater_is_better : bool
        Is more better?
    scorer : str, optional
        Name of scorer
    ret_index : bool
        Return index or by?


    Return
    ------
    Function
        A function that take as input `cv_results`: dict of numpy(masked)
        ndarrays (See attribute cv_results_ of `GridSearchCV`), and returns
        either the index of a model with the least complex parameters (by), or
        the parameters, depending on ret_index
    """

    if scorer is None:
        scorer = "score"

    def best_low_complexity(cv_results):
        import pandas as pd
        import numpy as np

        cv_results = pd.DataFrame(cv_results)

        threshold = se_bound(
            cv_results,
            greater_is_better=greater_is_better,
            scorer=scorer,
            n_fold=n_fold,
        )

        if greater_is_better:
            candidate_idx = np.flatnonzero(
                cv_results[f"mean_test_{scorer}"] >= threshold
            )
        else:
            candidate_idx = np.flatnonzero(
                cv_results[f"mean_test_{scorer}"] <= threshold
            )

        cv_results["___index"] = np.arange(len(cv_results))

        cv_least_complex = (
            cv_results.loc[candidate_idx, :]
            .sort_values(by=by, ascending=ascending)
            .reset_index()
        )

        if ret_index:
            return cv_least_complex.loc[0, "___index"]
        else:
            return cv_least_complex.loc[[0], by]

    return best_low_complexity
