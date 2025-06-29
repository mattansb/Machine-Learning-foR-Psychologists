

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)

<sub>*Last updated 2025-06-25.*</sub>

This Github repo contains all lesson files for *Machine Learning in R*.
The goal is to impart students with the basic tools to construct,
evaluate and compare various **machine learning models, using
[`tidymodels`](https://www.tidymodels.org/)**. (Prior to 2025, this
course was based on the `caret` package - this version can still be
found
[here](https://github.com/mattansb/Machine-Learning-foR-Psychologists/tree/caret).)

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
    (preferably version 4.3.2 or above).
2.  [RStudio IDE](https://www.rstudio.com/products/rstudio/download/)
    (optional, but recommended).
3.  The following packages, listed by lesson:

| Lesson | Packages |
|----|----|
| [01 Intro with Regression](/01%20Intro%20with%20Regression) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`patchwork`](https://CRAN.R-project.org/package=patchwork), [`kknn`](https://CRAN.R-project.org/package=kknn) |
| [02 Classification](/02%20Classification) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`parameters`](https://CRAN.R-project.org/package=parameters), [`kknn`](https://CRAN.R-project.org/package=kknn), [`palmerpenguins`](https://CRAN.R-project.org/package=palmerpenguins), [`themis`](https://CRAN.R-project.org/package=themis), [`patchwork`](https://CRAN.R-project.org/package=patchwork), [`probably`](https://CRAN.R-project.org/package=probably), [`tailor`](https://CRAN.R-project.org/package=tailor), [`modeldata`](https://CRAN.R-project.org/package=modeldata) |
| [03 Resampling and Tuning](/03%20Resampling%20and%20Tuning) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kknn`](https://CRAN.R-project.org/package=kknn), [`finetune`](https://CRAN.R-project.org/package=finetune), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`glue`](https://CRAN.R-project.org/package=glue), [`modeldata`](https://CRAN.R-project.org/package=modeldata), [`patchwork`](https://CRAN.R-project.org/package=patchwork), [`datasets`](https://CRAN.R-project.org/package=datasets), [`tune`](https://CRAN.R-project.org/package=tune), [`performance`](https://CRAN.R-project.org/package=performance) |
| [04 The problem of over-dimensionality](/04%20The%20problem%20of%20over-dimensionality) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`leaps`](https://CRAN.R-project.org/package=leaps), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`MASS`](https://CRAN.R-project.org/package=MASS), [`insight`](https://CRAN.R-project.org/package=insight), [`glmnet`](https://CRAN.R-project.org/package=glmnet), [`tidyr`](https://CRAN.R-project.org/package=tidyr), [`scales`](https://CRAN.R-project.org/package=scales), [`tibble`](https://CRAN.R-project.org/package=tibble), [`ggplot2`](https://CRAN.R-project.org/package=ggplot2), [`vip`](https://CRAN.R-project.org/package=vip), [`kknn`](https://CRAN.R-project.org/package=kknn), [`plsmod`](https://CRAN.R-project.org/package=plsmod), [`BiocManager`](https://CRAN.R-project.org/package=BiocManager) |
| [05 SVM](/05%20SVM) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kernlab`](https://CRAN.R-project.org/package=kernlab), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`palmerpenguins`](https://CRAN.R-project.org/package=palmerpenguins) |
| [06 Trees](/06%20Trees) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`rpart`](https://CRAN.R-project.org/package=rpart), [`rpart.plot`](https://CRAN.R-project.org/package=rpart.plot), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`scales`](https://CRAN.R-project.org/package=scales), [`vip`](https://CRAN.R-project.org/package=vip), [`MASS`](https://CRAN.R-project.org/package=MASS), [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`baguette`](https://CRAN.R-project.org/package=baguette), [`randomForest`](https://CRAN.R-project.org/package=randomForest), [`xgboost`](https://CRAN.R-project.org/package=xgboost), [`MASS`](https://CRAN.R-project.org/package=MASS), [`forcats`](https://CRAN.R-project.org/package=forcats), [`vip`](https://CRAN.R-project.org/package=vip) |
| [07 explain predictions](/07%20explain%20predictions) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kknn`](https://CRAN.R-project.org/package=kknn), [`randomForest`](https://CRAN.R-project.org/package=randomForest), [`patchwork`](https://CRAN.R-project.org/package=patchwork), [`DALEX`](https://CRAN.R-project.org/package=DALEX), [`DALEXtra`](https://CRAN.R-project.org/package=DALEXtra), [`marginaleffects`](https://CRAN.R-project.org/package=marginaleffects), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`palmerpenguins`](https://CRAN.R-project.org/package=palmerpenguins), [`vip`](https://CRAN.R-project.org/package=vip) |
| [08 unsupervised learning](/08%20unsupervised%20learning) | [`tidyverse`](https://CRAN.R-project.org/package=tidyverse), [`patchwork`](https://CRAN.R-project.org/package=patchwork), [`recipes`](https://CRAN.R-project.org/package=recipes), [`Rtsne`](https://CRAN.R-project.org/package=Rtsne), [`factoextra`](https://CRAN.R-project.org/package=factoextra), [`ggrepel`](https://CRAN.R-project.org/package=ggrepel), [`cluster`](https://CRAN.R-project.org/package=cluster), [`randomForest`](https://CRAN.R-project.org/package=randomForest), [`modeldata`](https://CRAN.R-project.org/package=modeldata), [`psych`](https://CRAN.R-project.org/package=psych), [`parameters`](https://CRAN.R-project.org/package=parameters), [`performance`](https://CRAN.R-project.org/package=performance), [`dplyr`](https://CRAN.R-project.org/package=dplyr), [`tidyr`](https://CRAN.R-project.org/package=tidyr), [`datasets`](https://CRAN.R-project.org/package=datasets), [`GPArotation`](https://CRAN.R-project.org/package=GPArotation), [`psychTools`](https://CRAN.R-project.org/package=psychTools) |

You can install all the packages used by running:

    # in alphabetical order:

    pkgs <- c(
      "baguette", "BiocManager", "cluster", "DALEX", "DALEXtra",
      "datasets", "dplyr", "factoextra", "finetune", "forcats", "ggplot2",
      "ggrepel", "glmnet", "glue", "GPArotation", "insight", "ISLR",
      "kernlab", "kknn", "leaps", "marginaleffects", "MASS", "modeldata",
      "palmerpenguins", "parameters", "patchwork", "performance", "plsmod",
      "probably", "psych", "psychTools", "randomForest", "recipes",
      "rpart", "rpart.plot", "Rtsne", "scales", "stats", "tailor",
      "themis", "tibble", "tidymodels", "tidyr", "tidyverse", "tune",
      "vip", "xgboost"
    )

    install.packages(pkgs, dependencies = TRUE)

<details>
<summary>
<i>Package Versions</i>
</summary>

The package versions used here:

- `baguette` 1.1.0 (*CRAN*)
- `BiocManager` 1.30.25 (*CRAN*)
- `cluster` 2.1.6 (*CRAN*)
- `DALEX` 2.4.3 (*CRAN*)
- `DALEXtra` 2.3.0 (*CRAN*)
- `datasets` 4.4.1 (*Dev*)
- `dplyr` 1.1.4 (*CRAN*)
- `factoextra` 1.0.7 (*CRAN*)
- `finetune` 1.2.0 (*CRAN*)
- `forcats` 1.0.0 (*CRAN*)
- `ggplot2` 3.5.1 (*CRAN*)
- `ggrepel` 0.9.6 (*CRAN*)
- `glmnet` 4.1-8 (*CRAN*)
- `glue` 1.8.0 (*CRAN*)
- `GPArotation` 2024.3-1 (*CRAN*)
- `insight` 1.3.0 (*CRAN*)
- `ISLR` 1.4 (*CRAN*)
- `kernlab` 0.9-33 (*CRAN*)
- `kknn` 1.3.1 (*CRAN*)
- `leaps` 3.2 (*CRAN*)
- `marginaleffects` 0.26.0.3 (*Github:
  vincentarelbundock/marginaleffects*)
- `MASS` 7.3-60.2 (*CRAN*)
- `modeldata` 1.4.0 (*CRAN*)
- `palmerpenguins` 0.1.1 (*CRAN*)
- `parameters` 0.26.0.1 (*Dev*)
- `patchwork` 1.3.0 (*CRAN*)
- `performance` 0.14.0 (*CRAN*)
- `plsmod` 1.0.0 (*CRAN*)
- `probably` 1.0.3 (*CRAN*)
- `psych` 2.4.12 (*CRAN*)
- `psychTools` 2.4.3 (*CRAN*)
- `randomForest` 4.7-1.2 (*CRAN*)
- `recipes` 1.1.0 (*CRAN*)
- `rpart` 4.1.23 (*CRAN*)
- `rpart.plot` 3.1.2 (*CRAN*)
- `Rtsne` 0.17 (*CRAN*)
- `scales` 1.3.0 (*CRAN*)
- `stats` 4.4.1 (*Dev*)
- `tailor` 0.0.0.9002 (*Github: tidymodels/tailor*)
- `themis` 1.0.3 (*CRAN*)
- `tibble` 3.2.1 (*CRAN*)
- `tidymodels` 1.2.0 (*CRAN*)
- `tidyr` 1.3.1 (*CRAN*)
- `tidyverse` 2.0.0 (*CRAN*)
- `tune` 1.2.1 (*CRAN*)
- `vip` 0.4.1 (*CRAN*)
- `xgboost` 1.7.8.1 (*CRAN*)

</details>
