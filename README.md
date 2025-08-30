

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)

<sub>*Last updated 2025-07-27.*</sub>

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
  in R/Python*, and does not contain any theoretical or introductory
  materials.
- Please note that some code does not work *on purpose*, to force
  students to learn to debug.

## Setup

You will need:

1.  A fresh installation of [**R**](https://cran.r-project.org/)
    (preferably version 4.4.1 or above)
2.  [RStudio](httpshttps://posit.co/download/rstudio-desktop/) *or*
    [Positron](https://positron.posit.co/download.html) (optional, but
    recommended).
3.  The following packages, listed by lesson:

| Lesson | Packages |
|:---|:---|
| [01 Intro with Regression](/01%20Intro%20with%20Regression) | `ISLR`, `tidymodels`, `stats`, `patchwork`, `kknn` |
| [02 Classification](/02%20Classification) | `tidymodels`, `ISLR`, `stats`, `parameters`, `kknn`, `palmerpenguins`, `themis`, `patchwork`, `probably`, `tailor`, `modeldata` |
| [03 Resampling and Tuning](/03%20Resampling%20and%20Tuning) | `tidymodels`, `kknn`, `finetune`, `ISLR`, `glue`, `modeldata`, `patchwork`, `datasets`, `tune`, `performance` |
| [04 The problem of over-dimensionality](/04%20The%20problem%20of%20over-dimensionality) | `tidymodels`, `leaps`, `ISLR`, `stats`, `MASS`, `insight`, `glmnet`, `tidyr`, `scales`, `tibble`, `ggplot2`, `vip`, `kknn`, `BiocManager`, `mixOmics`, `plsmod` |
| [05 SVM](/05%20SVM) | `tidymodels`, `kernlab`, `ISLR`, `palmerpenguins` |
| [06 Trees](/06%20Trees) | `tidymodels`, `rpart`, `rpart.plot`, `ISLR`, `scales`, `vip`, `MASS`, `tidymodels`, `baguette`, `randomForest`, `xgboost`, `MASS`, `forcats`, `vip` |
| [07 explain predictions](/07%20explain%20predictions) | `tidymodels`, `kknn`, `randomForest`, `patchwork`, `DALEX`, `DALEXtra`, `marginaleffects`, `ISLR`, `palmerpenguins`, `vip` |
| [08 unsupervised learning](/08%20unsupervised%20learning) | `tidyverse`, `patchwork`, `recipes`, `Rtsne`, `factoextra`, `ggrepel`, `cluster`, `randomForest`, `modeldata`, `psych`, `parameters`, `performance`, `datasets`, `GPArotation`, `dplyr`, `tidyr`, `psychTools` |

<details>
<summary>
<i>Installing R Packages</i>
</summary>

You can install all the R packages used by running:

    # in alphabetical order:

    pak::pak(
      c(

        "cran::BiocManager" # 1.30.25
        "cran::DALEX" # 2.4.3
        "cran::DALEXtra" # 2.3.0
        "cran::GPArotation" # 2024.3-1
        "cran::ISLR" # 1.4
        "cran::MASS" # 7.3-60.2
        "cran::Rtsne" # 0.17
        "cran::baguette" # 1.1.0
        "cran::cluster" # 2.1.6
        "cran::dplyr" # 1.1.4
        "cran::factoextra" # 1.0.7
        "cran::finetune" # 1.2.0
        "cran::forcats" # 1.0.0
        "cran::ggplot2" # 3.5.1
        "cran::ggrepel" # 0.9.6
        "cran::glmnet" # 4.1-8
        "cran::glue" # 1.8.0
        "cran::insight" # 1.3.1
        "cran::kernlab" # 0.9-33
        "cran::kknn" # 1.3.1
        "cran::leaps" # 3.2
        "github::vincentarelbundock/marginaleffects" # 0.26.0.3
        "bioc::mixOmics" # 6.30.0
        "cran::modeldata" # 1.4.0
        "cran::palmerpenguins" # 0.1.1
        "parameters" # 0.26.0.1
        "cran::patchwork" # 1.3.0
        "cran::performance" # 0.14.0
        "cran::plsmod" # 1.0.0
        "cran::probably" # 1.0.3
        "cran::psych" # 2.4.12
        "cran::psychTools" # 2.4.3
        "cran::randomForest" # 4.7-1.2
        "github::tidymodels/recipes" # 1.3.1.9000
        "cran::rpart" # 4.1.23
        "cran::rpart.plot" # 3.1.2
        "cran::scales" # 1.3.0
        "github::tidymodels/tailor" # 0.0.0.9002
        "cran::themis" # 1.0.3
        "cran::tibble" # 3.2.1
        "cran::tidymodels" # 1.2.0
        "cran::tidyr" # 1.3.1
        "cran::tidyverse" # 2.0.0
        "github::tidymodels/tune" # 1.3.0.9001
        "cran::vip" # 0.4.1
        "cran::xgboost" # 1.7.8.1

      )
    )

</details>

------------------------------------------------------------------------

Prior to 2025, this course was based on the `{caret}` package - this
version can still be found
[here](https://github.com/mattansb/Machine-Learning-foR-Psychologists/tree/caret).

An experimental Python version can be found [here](python).
