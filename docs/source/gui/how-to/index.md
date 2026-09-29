# How-to guides

One page per stage of the Bayesian workflow, written as things you click.

The workflow runs in this order. Some stages send you back to an earlier one.

| Stage | Page | Next |
| --- | --- | --- |
| a. Load data | {doc}`01-load-data`, {doc}`02-explore-data` | {doc}`03-specify-model` |
| b. Specify the model | {doc}`03-specify-model` | {doc}`04-check-priors` |
| c. Check the priors | {doc}`04-check-priors` | Reasonable: {doc}`05-fit-model`. Not reasonable: {doc}`03-specify-model` |
| d. Fit | {doc}`05-fit-model` | {doc}`06-check-convergence` |
| e. Check convergence | {doc}`06-check-convergence` | Converged: {doc}`08-check-fit-and-compare`. Not converged: {doc}`07-update-model` |
| e. Update the model | {doc}`07-update-model` | {doc}`06-check-convergence` |
| f. Check the fit and compare | {doc}`08-check-fit-and-compare` | Adequate: {doc}`09-prior-sensitivity`. Not adequate: {doc}`03-specify-model` |
| g. Check prior sensitivity | {doc}`09-prior-sensitivity` | {doc}`10-estimate-areas` |
| h. Estimate the areas | {doc}`10-estimate-areas` | Report the results |

```{toctree}
:maxdepth: 1
:hidden:

01-load-data
02-explore-data
03-specify-model
04-check-priors
05-fit-model
06-check-convergence
07-update-model
08-check-fit-and-compare
09-prior-sensitivity
10-estimate-areas
```