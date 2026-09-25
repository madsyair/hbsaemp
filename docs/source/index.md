# hbsaemp

Hierarchical Bayes small area estimation for Python, built on Bambi, PyMC and ArviZ.

There are two ways to use it, and the documentation is split the same way.
Pick one; each track is complete on its own.

## Install

hbsaemp is not on PyPI; install it straight from the GitHub repository.

```bash
pip install "hbsaemp[bambi] @ git+https://github.com/madsyair/hbsaemp"   # Python API
pip install "hbsaemp[gui] @ git+https://github.com/madsyair/hbsaemp"     # adds the graphical interface
```

## Choose a track

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: python/index
:link-type: doc

Write the analysis as code. Full control, reproducible scripts and notebooks.
:::

:::{grid-item-card} Graphical interface
:link: gui/index
:link-type: doc

Work through the analysis by clicking. No Python beyond starting the app.
:::
::::

## How each track is arranged

Every track holds the same four kinds of page.

- **Tutorials** take you through one whole analysis. Read them when you are new.
- **How-to guides** answer one stage at a time. Read them when you are working.
- **Reference** states technical facts. Read it when you need a fact.
- **Explanation** gives the reasoning. Read it when you want to understand.

## About the project

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item-card} Authors
:link: about/authors
:link-type: doc
Who develops hbsaemp.
:::

:::{grid-item-card} Citation
:link: about/citation
:link-type: doc
How to cite it.
:::

:::{grid-item-card} License
:link: about/license
:link-type: doc
GPL-3.0-or-later.
:::

:::{grid-item-card} References
:link: about/references
:link-type: doc
Works cited in these pages.
:::
::::

The documentation is organised following Diátaxis: Procida, D. (2022, 9 May).
*Diátaxis: A systematic approach to technical documentation authoring*.
<https://diataxis.fr/>

```{toctree}
:maxdepth: 2
:hidden:

python/index
gui/index
```
