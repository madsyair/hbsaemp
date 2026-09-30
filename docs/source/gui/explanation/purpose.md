# What the GUI is for

The graphical interface is one way into `hbsaemp`. It carries an analysis from a data
file to small-area estimates without any code.

## The question

Every function in the package can be called from Python. Why does the package also ship
a dashboard?

## Background

`hbsaemp` is the Python counterpart of the R package `hbsaems`. Its Python interface
asks the user to import the package, pick the function for the response family, pass
formulas and settings, and read the result objects. A practitioner who works in R has to
learn that interface before running a first model. The dashboard removes that
requirement. The single command that starts it is the only Python the GUI track needs.

## The reasoning

**The GUI lets a person run a Hierarchical Bayes SAE analysis by clicking.** Loading
data, choosing variables, checking priors, fitting, checking convergence, comparing
models, and estimating areas are all done in the browser. The
{doc}`how-to guides <../how-to/index>` follow that order, one page per stage.

**The layout builds the checks of the Bayesian workflow into the tabs.** The Modeling
tab puts the prior predictive check before Fit Model, and Results opens on Convergence
Evaluation. The tabs do not match the workflow order exactly, because the posterior
predictive check sits on the Modeling tab, before Results. The how-to guides follow the
workflow order instead. {doc}`convergence` explains one place where the app deliberately
does not enforce a check.

**The GUI and the Python API run the same code.** The Modeling tab builds a model with
the same `hbm_gaussian`, `hbm_beta`, or `hbm_binomial` function that a Python user
calls. The read-only code preview shows the equivalent Python code, and
**Save Code (.py)** writes it to a file. A model built by clicking can therefore be reproduced,
reviewed, and extended in Python.

**The app runs on the user's own computer.** It starts a local web server, and the
browser is only the screen. Uploaded files stay on the machine that runs the server,
unless someone deploys the server elsewhere. The upload limit defaults to 50 MB and is
set through `AppConfig.max_upload_mb`.

## What the GUI does not try to do

The GUI covers the beginner interface only. It offers the Gaussian, Beta, and Binomial
families, and it sets priors automatically. The only prior-related controls are
**Pin sigma** and **Pin kappa**. Custom priors, `hbm_flex`, and `create_model` are available
in Python. The {doc}`../../python/index` track covers them.

## Related

- {doc}`../how-to/index` for the stages in order.
- {doc}`results` for how the Results tab is organised.
- {doc}`convergence` for what saving a model does and does not check.
