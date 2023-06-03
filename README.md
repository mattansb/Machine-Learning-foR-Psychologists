
# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)  
[![](https://img.shields.io/badge/Language-R-blue)](http://cran.r-project.org/)

<sub>*Last updated 2023-06-03.*</sub>

This Github repo contains all lesson files for *Machine Learning in R*.
The goal is to impart students with the basic tools to construct,
evaluate and compare various **machine learning models, using
[`caret`](https://topepo.github.io/caret/)**. (Materials developed with
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
    (preferably version 4.2 or above).
2.  [RStudio IDE](https://www.rstudio.com/products/rstudio/download/)
    (optional, but recommended).
3.  The following packages, listed by lesson:

| Lesson                                                                                  | Packages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [01 Intro with Regression](/01%20Intro%20with%20Regression)                             | [`caret`](https://CRAN.R-project.org/package=caret), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`rsample`](https://CRAN.R-project.org/package=rsample), [`recipes`](https://CRAN.R-project.org/package=recipes), [`yardstick`](https://CRAN.R-project.org/package=yardstick)                                                                                                                                                                                                                                                                               |
| [02 Classification and Tuning](/02%20Classification%20and%20Tuning)                     | [`dplyr`](https://CRAN.R-project.org/package=dplyr), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`caret`](https://CRAN.R-project.org/package=caret), [`rsample`](https://CRAN.R-project.org/package=rsample), [`yardstick`](https://CRAN.R-project.org/package=yardstick), [`recipes`](https://CRAN.R-project.org/package=recipes), [`ggplot2`](https://CRAN.R-project.org/package=ggplot2), [`psych`](https://CRAN.R-project.org/package=psych)                                                                                                            |
| [03 The problem of over-dimensionality](/03%20The%20problem%20of%20over-dimensionality) | [`dplyr`](https://CRAN.R-project.org/package=dplyr), [`ggplot2`](https://CRAN.R-project.org/package=ggplot2), [`ISLR`](https://CRAN.R-project.org/package=ISLR), [`caret`](https://CRAN.R-project.org/package=caret), [`rsample`](https://CRAN.R-project.org/package=rsample), [`yardstick`](https://CRAN.R-project.org/package=yardstick), [`recipes`](https://CRAN.R-project.org/package=recipes), [`leaps`](https://CRAN.R-project.org/package=leaps), [`tidyr`](https://CRAN.R-project.org/package=tidyr), [`MASS`](https://CRAN.R-project.org/package=MASS)    |
| [04 Trees](/04%20Trees)                                                                 | [`dplyr`](https://CRAN.R-project.org/package=dplyr), [`rsample`](https://CRAN.R-project.org/package=rsample), [`recipes`](https://CRAN.R-project.org/package=recipes), [`caret`](https://CRAN.R-project.org/package=caret), [`yardstick`](https://CRAN.R-project.org/package=yardstick)                                                                                                                                                                                                                                                                             |
| [05 SVM](/05%20SVM)                                                                     | [`ggplot2`](https://CRAN.R-project.org/package=ggplot2), [`rsample`](https://CRAN.R-project.org/package=rsample), [`recipes`](https://CRAN.R-project.org/package=recipes), [`caret`](https://CRAN.R-project.org/package=caret), [`yardstick`](https://CRAN.R-project.org/package=yardstick), [`forcats`](https://CRAN.R-project.org/package=forcats)                                                                                                                                                                                                                |
| [06 unsupervised learning](/06%20unsupervised%20learning)                               | [`tidyverse`](https://CRAN.R-project.org/package=tidyverse), [`recipes`](https://CRAN.R-project.org/package=recipes), [`psych`](https://CRAN.R-project.org/package=psych), [`parameters`](https://CRAN.R-project.org/package=parameters), [`factoextra`](https://CRAN.R-project.org/package=factoextra), [`performance`](https://CRAN.R-project.org/package=performance), [`nFactors`](https://CRAN.R-project.org/package=nFactors), [`GPArotation`](https://CRAN.R-project.org/package=GPArotation), [`datawizard`](https://CRAN.R-project.org/package=datawizard) |
| [07 explain predictions](/07%20explain%20predictions)                                   | [`recipes`](https://CRAN.R-project.org/package=recipes), [`caret`](https://CRAN.R-project.org/package=caret), [`yardstick`](https://CRAN.R-project.org/package=yardstick), [`lime`](https://CRAN.R-project.org/package=lime), [`kernelshap`](https://CRAN.R-project.org/package=kernelshap), [`shapviz`](https://CRAN.R-project.org/package=shapviz), [`tidyr`](https://CRAN.R-project.org/package=tidyr), [`forcats`](https://CRAN.R-project.org/package=forcats)                                                                                                  |

You can install all the packages used by running:

    # in alphabetical order:

    pkgs <- c(
      "caret", "datawizard", "dplyr", "factoextra", "forcats", "ggplot2",
      "GPArotation", "ISLR", "kernelshap", "leaps", "lime", "MASS",
      "nFactors", "parameters", "performance", "psych", "recipes",
      "rsample", "shapviz", "tidyr", "tidyverse", "yardstick"
    )

    install.packages(pkgs, dependencies = TRUE)

<details>
<summary>
<i>Package Versions</i>
</summary>

The package versions used here:

- `caret` 6.0-94 (*CRAN*)
- `datawizard` 0.7.1 (*CRAN*)
- `dplyr` 1.1.1 (*CRAN*)
- `factoextra` 1.0.7 (*CRAN*)
- `forcats` 1.0.0 (*CRAN*)
- `ggplot2` 3.4.2 (*CRAN*)
- `GPArotation` 2023.3-1 (*CRAN*)
- `ISLR` 1.4 (*CRAN*)
- `kernelshap` 0.3.7 (*CRAN*)
- `leaps` 3.1 (*CRAN*)
- `lime` 0.5.3 (*CRAN*)
- `MASS` 7.3-58.1 (*CRAN*)
- `nFactors` 2.4.1.1 (*CRAN*)
- `parameters` 0.21.0 (*CRAN*)
- `performance` 0.10.3 (*CRAN*)
- `psych` 2.3.3 (*CRAN*)
- `recipes` 1.0.5 (*CRAN*)
- `rsample` 1.1.1 (*CRAN*)
- `shapviz` 0.8.0 (*CRAN*)
- `tidyr` 1.3.0 (*CRAN*)
- `tidyverse` 2.0.0 (*CRAN*)
- `yardstick` 1.1.0 (*CRAN*)

</details>
