

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)

<sub>*Last updated 2026-01-23.*</sub>

This Github repo contains all lesson files for *Machine Learning in R*.
The goal is to impart students with the basic tools to construct,
evaluate and compare various **machine learning models, using
[`tidymodels`](https://www.tidymodels.org/)**.

These topics were taught in the graduate-level course ***Machine
Learning for Psychologists*** (Psych Dep., Ben-Gurion University of the
Negev; Psych Dep., Tel-Aviv University). (Materials developed with Yael
Bar-Shachar.) This course assumes basic competence in R (importing,
regression modeling, plotting, etc.), along the lines of [*Practical
Applications in R for
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
| [01 intro with regression](/01%20intro%20with%20regression) | `ISLR`, `tidymodels`, `stats`, `patchwork`, `kknn` |
| [02 resampling](/02%20resampling) | `tidymodels`, `kknn`, `finetune`, `ISLR`, `mirai`, `glue`, `modeldata`, `patchwork`, `datasets`, `tune`, `performance` |
| [03 classification](/03%20classification) | `tidymodels`, `ISLR`, `stats`, `parameters`, `kknn`, `mirai`, `themis`, `patchwork`, `tailor`, `modeldata`, `probably`, `desirability2` |
| [04 the problem of over-dimensionality](/04%20the%20problem%20of%20over-dimensionality) | `tidymodels`, `leaps`, `ISLR`, `stats`, `MASS`, `insight`, `glmnet`, `mirai`, `tidyr`, `scales`, `tibble`, `ggplot2`, `vip`, `modeldata`, `kknn`, `BiocManager`, `mixOmics`, `parsnip` |
| [05 svm](/05%20svm) | `tidymodels`, `kernlab`, `mirai`, `ISLR`, `ISLR2` |
| [06 trees](/06%20trees) | `tidymodels`, `rpart`, `rpart.plot`, `mirai`, `ISLR`, `scales`, `vip`, `MASS`, `baguette`, `ranger`, `xgboost`, `forcats`, `randomForest` |
| [07 explanatory model analysis](/07%20explanatory%20model%20analysis) | `tidymodels`, `kknn`, `ranger`, `patchwork`, `DALEX`, `DALEXtra`, `marginaleffects`, `ISLR`, `datawizard`, `vip`, `randomForest` |
| [08 clustering](/08%20clustering) | `tidymodels`, `patchwork`, `cluster`, `Rtsne`, `factoextra`, `ggrepel`, `randomForest`, `fpc`, `pak`, `clusterpval`, `modeldata` |

<details>

<summary>

<i>Installing R Packages</i>
</summary>

You can install all the R packages used by running:

    # in alphabetical order:

    pak::pak(
      c(

        "cran::BiocManager", # 1.30.27
        "cran::DALEX", # 2.5.3
        "cran::DALEXtra", # 2.3.1
        "cran::ISLR", # 1.4
        "cran::ISLR2", # 1.3-2
        "cran::MASS", # 7.3-65
        "cran::Rtsne", # 0.17
        "cran::baguette", # 1.1.0
        "cran::cluster", # 2.1.8.1
        "github::lucylgao/clusterpval", # 1.0.1
        "datawizard", # 1.3.0
        "cran::desirability2", # 0.2.0
        "cran::factoextra", # 1.0.7
        "cran::finetune", # 1.2.1
        "cran::forcats", # 1.0.1
        "cran::fpc", # 2.2-14
        "cran::ggplot2", # 4.0.1
        "cran::ggrepel", # 0.9.6
        "cran::glmnet", # 4.1-10
        "cran::glue", # 1.8.0
        "insight", # 1.4.4.3
        "cran::kernlab", # 0.9-33
        "cran::kknn", # 1.4.1
        "cran::leaps", # 3.2
        "cran::marginaleffects", # 0.31.0
        "cran::mirai", # 2.5.3
        "mixOmics", # 6.34.0
        "cran::modeldata", # 1.5.1
        "cran::pak", # 0.9.2
        "parameters", # 0.28.3
        "cran::parsnip", # 1.4.1
        "cran::patchwork", # 1.3.2
        "cran::performance", # 0.15.3
        "cran::probably", # 1.2.0
        "cran::randomForest", # 4.7-1.2
        "cran::ranger", # 0.18.0
        "cran::rpart", # 4.1.24
        "cran::rpart.plot", # 3.1.4
        "cran::scales", # 1.4.0
        "cran::tailor", # 0.1.0
        "cran::themis", # 1.0.3
        "cran::tibble", # 3.3.1
        "cran::tidymodels", # 1.4.1
        "cran::tidyr", # 1.3.2
        "cran::tune", # 2.0.1
        "cran::vip", # 0.4.5
        "cran::xgboost" # 3.1.3.1

      )
    )

</details>

------------------------------------------------------------------------

Prior to 2025, this course was based on the `{caret}` package - this
version can still be found
[here](https://github.com/mattansb/Machine-Learning-foR-Psychologists/tree/caret).
