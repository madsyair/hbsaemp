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

```{toctree}
:maxdepth: 2
:hidden:

python/index
gui/index
```
