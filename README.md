

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
[![](https://img.shields.io/badge/Language-R-blue.png)](http://cran.r-project.org/)
[![](https://img.shields.io/badge/Language-Python-blue.png)](http://python.org/)

<sub>*Last updated 2026-11-24.*</sub>

This Github repo contains all lesson files for *Machine Learning in R
and Python*. The goal is to impart students with the basic tools to
construct, evaluate and compare various **machine learning models, using
[`tidymodels`](https://www.tidymodels.org/)[^1] and
[`scikit-learn`](https://scikit-learn.org/)**, based on [*An
Introduction to Statistical Learning*](https://www.statlearning.com/).

These topics are taught in the graduate-level course ***Machine Learning
for Psychologists*** (Psych Dep., Ben-Gurion University of the Negev;
Psych Dep., Tel-Aviv University). This course assumes basic competence
in R and Python (importing, regression modeling, plotting, etc.).[^2]

> [!NOTE]
>
> - This repo contains only materials relating to *Practical
>   Applications in R/Python*, and does not contain any theoretical or
>   introductory materials.
> - Please note that some code does not work *on purpose*, to force
>   students to learn to debug.

## Setup

You will need:

1.  A fresh installation of [**`R`**](https://cran.r-project.org/)
    (preferably 4.5.0+) and [**`Python`**](https://www.python.org/)
    (preferably 3.13+).
2.  [Positron](https://positron.posit.co/download)[^3].
3.  The following packages and modules, listed by lesson:

| Lesson | R Packages | Python Modules |
|:---|:---|:---|
| [01 intro with regression](01%20intro%20with%20regression//) | `ISLR`, `tidymodels`, `stats`, `patchwork`, `kknn` | `numpy`, `pandas`, `plotnine`, `patsy`, `statsmodels`, `scipy`, `ISLP`, `sklearn` |
| [02 cross validation](02%20cross%20validation//) | `tidymodels`, `kknn`, `finetune`, `ISLR`, `mirai`, `glue`, `modeldata`, `patchwork`, `datasets`, `tune`, `performance` | `pandas`, `plotnine`, `sklearn`, `ISLP`, `numpy` |
| [03 classification](03%20classification//) | `tidymodels`, `ISLR`, `stats`, `parameters`, `kknn`, `mirai`, `pak`, `MSBMisc`, `themis`, `patchwork`, `modeldata`, `probably`, `desirability2` | `numpy`, `pandas`, `plotnine`, `sklearn`, `ISLP`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `palmerpenguins`, `pandas`, `plotnine`, `sklearn`, `imblearn`, `ISLP`, `numpy`, `pandas`, `plotnine`, `sklearn`, `ISLP` |
| [04 the problem of over-dimensionality](04%20the%20problem%20of%20over-dimensionality//) | `stats`, `glmnet`, `tibble`, `ggplot2`, `hardhat`, `tidymodels`, `leaps`, `ISLR`, `MASS`, `insight`, `mirai`, `tidyr`, `vip`, `modeldata`, `kknn`, `mixOmics`, `pak`, `parsnip` |  |
| [05 svm](05%20svm//) | `tidymodels`, `kernlab`, `mirai`, `ISLR`, `ISLR2` |  |
| [06 trees](06%20trees//) | `tidymodels`, `rpart`, `rpart.plot`, `mirai`, `ISLR`, `scales`, `vip`, `MASS`, `baguette`, `ranger`, `xgboost`, `forcats`, `randomForest`, `glue` |  |
| [07 explanatory model analysis](07%20explanatory%20model%20analysis//) | `tidymodels`, `tailor`, `kknn`, `ranger`, `patchwork`, `DALEX`, `DALEXtra`, `marginaleffects`, `ISLR`, `datawizard`, `vip`, `randomForest` | `numpy`, `pandas`, `matplotlib`, `plotnine`, `sklearn`, `dalex`, `ISLP`, `palmerpenguins` |
| [08 clustering](08%20clustering//) | `tidymodels`, `tidyclust`, `MSBMisc`, `tibble`, `Rtsne`, `ggrepel`, `philentropy`, `randomForest`, `cluster`, `fpc`, `pak`, `clusterpval`, `modeldata` | `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotnine`, `sklearn`, `scipy`, `palmerpenguins` |

<details>

<summary>

<i>Installing R Packages ®️</i>
</summary>

You can install all the R packages used by running:

    # in alphabetical order:

    pak::pak(
      c(
        "baguette", # 1.1.0
        "cluster", # 2.1.8.3
        "github::github::lucylgao/clusterpval", # 1.0.1
        "DALEX", # 2.5.4
        "DALEXtra", # 2.3.1
        "NA", # 0.7.6
        "finetune", # 1.3.0
        "ggrepel", # 0.9.8
        "glmnet", # 5.1
        "glue", # 1.8.1
        "ISLR", # 1.4
        "kknn", # 1.4.1
        "leaps", # 3.2
        "marginaleffects", # 1.0.0
        "MASS", # 7.3-66
        "mirai", # 2.7.3
        "mixOmics", # 6.36.0
        "github::github::mattansb/MSBMisc", # 0.0.1.15
        "pak", # 0.11.1
        "patchwork", # 1.3.2
        "probably", # 1.2.0
        "randomForest", # 4.7-1.2
        "ranger", # 0.18.0
        "rpart.plot", # 3.1.5
        "scales", # 1.4.0
        "tidymodels", # 1.5.0
        "tidyverse", # 2.0.0
        "NA", # 0.5.0
        "xgboost" # 3.2.1.1
      )
    )

</details>

<details>

<summary>

<i>Installing Python Modules 🐍</i>
</summary>

You can install all the Python modules with the help of the
[`requirements.txt`](requirements.txt) file:

    ISLP>=0.4.0
    dalex>=1.8.0
    imbalanced-learn>=0.13.0
    matplotlib>=3.10.3
    numpy>=2.3.1
    palmerpenguins>=0.1.4
    pandas>=3.0.0
    patsy>=1.0.1
    plotnine>=0.15.0
    scikit-learn>=1.6.1
    scipy>=1.16.0
    seaborn>=0.13.2
    statsmodels>=0.14.5

And then running

    pip install -r requirements.txt

</details>

------------------------------------------------------------------------

### Acknowledgements

R materials were developed with Yael Bar-Shachar.

[^1]: Prior to 2025, this course was based on the
    [`{caret}`](https://topepo.github.io/caret/index.html) package -
    this version can still be found
    [here](https://github.com/mattansb/Machine-Learning-foR-Psychologists/tree/caret).

[^2]: For R, this mean something along the lines of [*Practical
    Applications in R for
    Psychologists*](https://github.com/mattansb/Practical-Applications-in-R-for-Psychologists).

[^3]: It is also possible to use
    [RStudio](https://www.rstudio.com/products/rstudio/download/), but
    Positron is preferred.
