

# Machine Learning foR Psychologists

[![](https://img.shields.io/badge/Open%20Educational%20Resources-Compatable-brightgreen.png)](https://creativecommons.org/about/program-areas/education-oer/)
[![](https://img.shields.io/badge/CC-BY--NC%204.0-lightgray)](http://creativecommons.org/licenses/by-nc/4.0/)
![](https://img.shields.io/badge/Languages-Python-blue.png)

<sub>*Last updated 2025-12-02.*</sub>

This folder contains Python translations of the R tutorials in the main
repo.

## Setup

You will need:

1.  A fresh installation of
    [**Python**](https://www.python.org/downloads/) (preferably version
    3.13 or above).
2.  [Positron IDE](https://positron.posit.co/download.html) (optional,
    but recommended).
3.  The following modules, listed by lesson:

| Lesson | Modules |
|:---|:---|
| [01 Intro with Regression](/01%20Intro%20with%20Regression) | `numpy`, `pandas`, `plotnine`, `patsy`, `statsmodels`, `scipy`, `ISLP`, `sklearn` |
| [02 Resampling and Tuning](/02%20Resampling%20and%20Tuning) | `pandas`, `plotnine`, `sklearn`, `ISLP`, `numpy` |

<details>

<summary>

<i>Installing Python Modules</i>
</summary>

You can install all the Python modules used by saving a
`requirements.txt` file:

    ISLP>=0.4.0
    numpy>=2.3.1
    pandas>=2.3.0
    patsy>=1.0.1
    plotnine>=0.15.0
    scikit-learn>=1.6.1
    scipy>=1.16.0
    statsmodels>=0.14.5

And then running

    pip install -r requirements.txt

</details>
