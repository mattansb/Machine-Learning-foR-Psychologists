
# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)  
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)

<sub>*Last updated 2024-08-21.*</sub>

This Github repo contains all lesson files for *Machine Learning in R*.
The goal is to impart students with the basic tools to construct,
evaluate and compare various **machine learning models, using
[`tidymodels`](https://www.tidymodels.org/)**. (Materials developed with
Yael Bar-Shachar.)

These topics were taught in the graduate-level course ***Machine
Learning for Psychologists*** (Psych Dep., Ben-Gurion University of the
Negev; Psych Dep., Tel-Aviv University). This course assumes basic
competence in R (importing, regression modeling, plotting, etc.), along
the lines of [*Practical Applications in R for
Psychologists*](https://github.com/mattansb/Practical-Applications-in-R-for-Psychologists).

**Notes:**

- This repo contains only materials relating to *Practical Applications
  in R*, and does not contain any theoretical or introductory
  materials.  
- Please note that some code does not work *on purpose*, to force
  students to learn to debug.

## Setup

You will need:

1.  A fresh installation of [**`R`**](https://cran.r-project.org/)
    (preferably version 4.3.2 or above).
2.  [RStudio IDE](https://www.rstudio.com/products/rstudio/download/)
    (optional, but recommended).
3.  The following packages, listed by lesson:

| Lesson                                                                                  | Packages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01 Intro with Regression](/01%20Intro%20with%20Regression)                             | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kknn`](https://CRAN.R-project.org/package=kknn), [`ISLR`](https://CRAN.R-project.org/package=ISLR)                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [02 Classification](/02%20Classification)                                               | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`parameters`](https://CRAN.R-project.org/package=parameters), [`kknn`](https://CRAN.R-project.org/package=kknn), [`palmerpenguins`](https://CRAN.R-project.org/package=palmerpenguins), [`themis`](https://CRAN.R-project.org/package=themis), [`probably`](https://CRAN.R-project.org/package=probably)                                                                                                                                                                                                                                                                                                               |
| [03 Resampling and Tuning](/03%20Resampling%20and%20Tuning)                             | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kknn`](https://CRAN.R-project.org/package=kknn), [`finetune`](https://CRAN.R-project.org/package=finetune), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`glue`](https://CRAN.R-project.org/package=glue), [`modeldata`](https://CRAN.R-project.org/package=modeldata)                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [04 The problem of over-dimensionality](/04%20The%20problem%20of%20over-dimensionality) | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`leaps`](https://CRAN.R-project.org/package=leaps), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`stats`](https://CRAN.R-project.org/package=stats), [`MASS`](https://CRAN.R-project.org/package=MASS), [`insight`](https://CRAN.R-project.org/package=insight), [`glmnet`](https://CRAN.R-project.org/package=glmnet), [`tidyr`](https://CRAN.R-project.org/package=tidyr), [`tibble`](https://CRAN.R-project.org/package=tibble), [`ggplot2`](https://CRAN.R-project.org/package=ggplot2), [`kknn`](https://CRAN.R-project.org/package=kknn), [`plsmod`](https://CRAN.R-project.org/package=plsmod), [`pls`](https://CRAN.R-project.org/package=pls), [`BiocManager`](https://CRAN.R-project.org/package=BiocManager) |
| [05 SVM](/05%20SVM)                                                                     | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kernlab`](https://CRAN.R-project.org/package=kernlab), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`palmerpenguins`](https://CRAN.R-project.org/package=palmerpenguins)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [06 Trees](/06%20Trees)                                                                 | [`dplyr`](https://CRAN.R-project.org/package=dplyr), [`rsample`](https://CRAN.R-project.org/package=rsample), [`recipes`](https://CRAN.R-project.org/package=recipes), [`caret`](https://CRAN.R-project.org/package=caret), [`yardstick`](https://CRAN.R-project.org/package=yardstick)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [07 explain predictions](/07%20explain%20predictions)                                   | [`tidymodels`](https://CRAN.R-project.org/package=tidymodels), [`kernelshap`](https://CRAN.R-project.org/package=kernelshap), [`shapviz`](https://CRAN.R-project.org/package=shapviz), [`ISLR`](https://CRAN.R-project.org/package=ISLR)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [08 unsupervised learning](/08%20unsupervised%20learning)                               | [`tidyverse`](https://CRAN.R-project.org/package=tidyverse), [`recipes`](https://CRAN.R-project.org/package=recipes), [`psych`](https://CRAN.R-project.org/package=psych), [`parameters`](https://CRAN.R-project.org/package=parameters), [`factoextra`](https://CRAN.R-project.org/package=factoextra), [`performance`](https://CRAN.R-project.org/package=performance), [`nFactors`](https://CRAN.R-project.org/package=nFactors), [`GPArotation`](https://CRAN.R-project.org/package=GPArotation), [`Rtsne`](https://CRAN.R-project.org/package=Rtsne), [`datawizard`](https://CRAN.R-project.org/package=datawizard), [`ggrepel`](https://CRAN.R-project.org/package=ggrepel)                                                                                                              |

You can install all the packages used by running:

    # in alphabetical order:

    pkgs <- c(
      "BiocManager", "caret", "datawizard", "dplyr", "factoextra",
      "finetune", "ggplot2", "ggrepel", "glmnet", "glue", "GPArotation",
      "insight", "ISLR", "kernelshap", "kernlab", "kknn", "leaps",
      "MASS", "modeldata", "nFactors", "palmerpenguins", "parameters",
      "performance", "pls", "plsmod", "probably", "psych", "recipes",
      "rsample", "Rtsne", "shapviz", "stats", "themis", "tibble", "tidymodels",
      "tidyr", "tidyverse", "yardstick"
    )

    install.packages(pkgs, dependencies = TRUE)

<details>
<summary>
<i>Package Versions</i>
</summary>

The package versions used here:

``` r
# | echo: false
packinfo <- installed.packages(fields = c("Package", "Version"))

get_src <- function(pkg) {
  pd <- packageDescription(pkg)
  if (is.null(src <- pd$Repository)) {
    if (!is.null(src <- pd$GithubRepo)) {
      src <- paste0("Github: ",pd$GithubUsername,"/",src)
    } else {
      src <- "Dev"
    }
  }
  return(src)
}

V <- packinfo[pkgs,"Version"]
src <- sapply(pkgs, get_src)
# setNames(paste0(V, " (", src,")"), pkgs)

v_info <- paste0(glue::glue(" - `{pkgs}` {V} (*{src}*)"), collapse = "\n")
```

- `BiocManager` 1.30.23 (*CRAN*)
- `caret` 6.0-94 (*CRAN*)
- `datawizard` 0.11.0 (*CRAN*)
- `dplyr` 1.1.4 (*CRAN*)
- `factoextra` 1.0.7 (*CRAN*)
- `finetune` 1.2.0 (*CRAN*)
- `ggplot2` 3.5.1 (*CRAN*)
- `ggrepel` 0.9.5 (*CRAN*)
- `glmnet` 4.1-8 (*CRAN*)
- `glue` 1.7.0 (*CRAN*)
- `GPArotation` 2024.3-1 (*CRAN*)
- `insight` 0.20.1 (*CRAN*)
- `ISLR` 1.4 (*CRAN*)
- `kernelshap` 0.4.1 (*CRAN*)
- `kernlab` 0.9-32 (*CRAN*)
- `kknn` 1.3.1 (*CRAN*)
- `leaps` 3.1 (*CRAN*)
- `MASS` 7.3-60.0.1 (*CRAN*)
- `modeldata` 1.3.0 (*CRAN*)
- `nFactors` 2.4.1.1 (*CRAN*)
- `palmerpenguins` 0.1.1 (*CRAN*)
- `parameters` 0.22.0 (*CRAN*)
- `performance` 0.12.0 (*CRAN*)
- `pls` 2.8-3 (*CRAN*)
- `plsmod` 1.0.0 (*CRAN*)
- `probably` 1.0.3 (*CRAN*)
- `psych` 2.4.3 (*CRAN*)
- `recipes` 1.0.10 (*CRAN*)
- `rsample` 1.2.1 (*CRAN*)
- `Rtsne` 0.17 (*CRAN*)
- `shapviz` 0.9.3 (*CRAN*)
- `stats` 4.3.2 (*Dev*)
- `themis` 1.0.2 (*CRAN*)
- `tibble` 3.2.1 (*CRAN*)
- `tidymodels` 1.2.0 (*CRAN*)
- `tidyr` 1.3.1 (*CRAN*)
- `tidyverse` 2.0.0 (*CRAN*)
- `yardstick` 1.3.1 (*CRAN*)

</details>
