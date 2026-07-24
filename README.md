

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)

<sub>*Last updated 2026-07-24.*</sub>

This Github repo contains all lesson files for *Machine Learning in R*.
The goal is to impart students with the basic tools to construct,
evaluate and compare various **machine learning models, using
[`tidymodels`](https://www.tidymodels.org/)**, based on [*An
Introduction to Statistical Learning: with applications in
R*](https://www.statlearning.com/).

These topics were taught in the graduate-level course ***Machine
Learning for Psychologists*** (Psych Dep., Ben-Gurion University of the
Negev; Psych Dep., Tel-Aviv University). This course assumes basic
competence in R (importing, regression modeling, plotting, etc.), along
the lines of [*Practical Applications in R for
Psychologists*](https://github.com/mattansb/Practical-Applications-in-R-for-Psychologists).

**Notes:**

- This repo contains only materials relating to *Practical Applications
  in R*, and does not contain any theoretical or introductory materials.
- Please note that some code does not work *on purpose*, to force
  students to learn to debug.

## Setup

You will need:

1.  A fresh installation of [**`R`**](https://cran.r-project.org/)
    (preferably version 4.5.0 or above).
2.  [RStudio IDE](https://www.rstudio.com/products/rstudio/download/) or
    [Positron](https://positron.posit.co/download) (optional, but
    recommended).
3.  The following packages, listed by lesson:

| Lesson | Packages |
|:---|:---|
| [01 intro with regression](01%20intro%20with%20regression//) | `ISLR`, `tidymodels`, `stats`, `patchwork`, `kknn` |
| [02 cross validation](02%20cross%20validation//) | `tidymodels`, `kknn`, `finetune`, `ISLR`, `mirai`, `glue`, `modeldata`, `patchwork`, `datasets`, `tune`, `performance` |
| [03 classification](03%20classification//) | `tidymodels`, `ISLR`, `stats`, `parameters`, `kknn`, `mirai`, `pak`, `MSBMisc`, `themis`, `patchwork`, `modeldata`, `probably`, `desirability2` |
| [04 the problem of over-dimensionality](04%20the%20problem%20of%20over-dimensionality//) | `stats`, `glmnet`, `tibble`, `ggplot2`, `hardhat`, `tidymodels`, `leaps`, `ISLR`, `MASS`, `insight`, `mirai`, `tidyr`, `vip`, `modeldata`, `kknn`, `mixOmics`, `pak`, `parsnip` |
| [05 svm](05%20svm//) | `tidymodels`, `kernlab`, `mirai`, `ISLR`, `ISLR2` |
| [06 trees](06%20trees//) | `tidymodels`, `rpart`, `rpart.plot`, `mirai`, `ISLR`, `scales`, `vip`, `MASS`, `baguette`, `ranger`, `xgboost`, `forcats`, `randomForest` |
| [07 explanatory model analysis](07%20explanatory%20model%20analysis//) | `tidymodels`, `kknn`, `ranger`, `patchwork`, `DALEX`, `DALEXtra`, `marginaleffects`, `ISLR`, `datawizard`, `vip`, `randomForest` |
| [08 clustering](08%20clustering//) | `tidymodels`, `tidyclust`, `MSBMisc`, `tibble`, `Rtsne`, `ggrepel`, `philentropy`, `randomForest`, `cluster`, `fpc`, `pak`, `clusterpval`, `modeldata` |

<details>

<summary>

<i>Installing R Packages</i>
</summary>

You can install all the R packages used by running:

    # in alphabetical order:

    pak::pak(
      c(

        "baguette", # 1.1.0
        "cluster", # 2.1.8.2
        "github::lucylgao/clusterpval", # 1.0.1
        "DALEX", # 2.5.3
        "DALEXtra", # 2.3.1
        "desirability2", # 0.2.0
        "easystats", # 0.7.5
        "finetune", # 1.2.1
        "fpc", # 2.2-14
        "ggrepel", # 0.9.7
        "glmnet", # 4.1-10
        "glue", # 1.8.0
        "ISLR", # 1.4
        "ISLR2", # 1.3-2
        "kernlab", # 0.9-33
        "kknn", # 1.4.1
        "leaps", # 3.2
        "github::vincentarelbundock/marginaleffects/r", # 0.32.0.5
        "mirai", # 2.6.1
        "mixOmics", # 6.34.0
        "github::mattansb/MSBMisc", # 0.0.1.15
        "pak", # 0.9.2
        "patchwork", # 1.3.2
        "philentropy", # 0.10.0
        "probably", # 1.2.0
        "randomForest", # 4.7-1.2
        "ranger", # 0.18.0
        "rpart", # 4.1.24
        "rpart.plot", # 3.1.4
        "Rtsne", # 0.17
        "scales", # 1.4.0
        "themis", # 1.0.3
        "tidyclust", # 0.3.1
        "tidymodels", # 1.4.1
        "tidyverse", # 2.0.0
        "vip", # 0.4.5
        "xgboost", # 3.2.0.1
        "cluster", # 2.1.8.1
        "MASS", # 7.3-65
        "rpart" # 4.1.24

      )
    )

</details>

------------------------------------------------------------------------

### Additional Versions

- Prior to 2025, this course was based on the `{caret}` package - this
  version can still be found
  [here](https://github.com/mattansb/Machine-Learning-foR-Psychologists/tree/caret).

- Partial parallel `python` lessons can be found in the [`py`
  folder](py/).

------------------------------------------------------------------------

### Acknowledgements

Materials developed with Yael Bar-Shachar.
