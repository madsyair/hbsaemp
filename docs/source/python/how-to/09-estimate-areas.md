---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 9 · Estimate the areas

Turn the chosen model into per-area estimates with their uncertainty.

## Before you start

You need the model that passed {doc}`05-check-convergence`, {doc}`07-check-fit-and-compare`
and {doc}`08-prior-sensitivity`:

```{code-cell} ipython3
from hbsaemp import create_model, estimate_areas, load_dataset, ModelConfig

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                     sampling_var="D", config=ModelConfig(target_accept=0.95,
                                                          progressbar=False))
model.fit()
```

## Do it

```{code-cell} ipython3
est = estimate_areas(model)
print(est.summary())
est.result_table.head()
```

The R-style alias `hbsae` is the same function. `est.mean_rse` and `est.mean_mse` give the
averages across areas.

Change the interval width with `ci_prob` (default 0.95):

```{code-cell} ipython3
est = estimate_areas(model, ci_prob=0.90)
```

### Areas without survey data

Pass `new_data=` to estimate areas outside the fitting data. The frame needs only the
auxiliary variables and the area column. An area label not seen during fitting is a
non-sampled area: at each posterior draw, its random effect is taken from a randomly chosen
fitted area, as brms does with `sample_new_levels = "uncertainty"`. Its interval is
therefore wider. The draws use the model's `random_seed`, so repeated calls agree.

## Check it worked

One row per area, with these columns:

| Column | Meaning |
| --- | --- |
| `mean`, `sd` | Posterior mean and standard deviation of the area quantity |
| `mse`, `rmse` | Posterior variance and its square root |
| `rse_pct` | Relative standard error, `sd / \|mean\| × 100` |
| `ci_lower`, `ci_upper` | Highest-density interval at `ci_prob` |

If an area label repeats in the data, the model was fitted on unit-level rows. The table is
then one row per observation, and a warning is logged.

**Shrinkage.** Compare `mean` with the direct estimate. A model-based estimate lies between
the direct estimate and the value predicted by the auxiliary variables, and moves furthest
where the sampling variance is largest. If every estimate equals its direct estimate, the
model adds nothing; recheck {doc}`02-specify-model`.

**RSE.** `rse_pct` is the usual criterion for publication. Report areas whose RSE is still
high with that caveat.

```{note}
The estimates summarise the posterior of the area means, not the posterior predictive
distribution. {doc}`../explanation/hb-sae` explains why.
```

This is the last stage of the workflow.

## If it fails

**`ModelNotFittedError`.** Fit the model first. This error is not wrapped in
`EstimationError`.

**`DataValidationError` naming a column.** `new_data` lacks an auxiliary variable or the
area column, or every row is missing a value in one of them.

**`EstimationError`.** The message gives the family, the shape of `new_data` and `ci_prob`.

**`ValueError` about `ci_prob`.** It must lie strictly between 0 and 1.

**Fewer rows than your input.** Rows with missing values are dropped before estimation.
Compare with `model.check_data()`, not with the raw DataFrame. For `new_data`, only a
missing auxiliary or area value drops a row.

## Related

Why small-area estimates are pulled towards the model: {doc}`../explanation/hb-sae`.
