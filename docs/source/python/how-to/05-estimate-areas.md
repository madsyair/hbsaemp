# 5 · Estimate the areas

Turn a fitted model into a table of per-area estimates with their uncertainty.

## Before you start

You need a fitted model that has passed {doc}`03-check-convergence`. This is the step
whose output you will actually publish, so everything upstream has to be trustworthy
before you read it.

```{testcode}
from hbsaemp import (create_model, load_dataset, estimate_areas,
                     ModelNotFittedError)

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")

try:
    estimate_areas(model)
except ModelNotFittedError as err:
    print(err.message)
```

```{testoutput}
Call GaussianModel.fit() before accessing .result.
```

## Do it

```{testsetup} fitted
from hbsaemp import create_model, load_dataset

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")
model.fit()
```

```{testcode} fitted
from hbsaemp import estimate_areas

est = estimate_areas(model)
print(est.summary())
print(est.result_table.head())
```

```{testoutput} fitted
:hide:
:options: +ELLIPSIS

AreaEstimatesResult
  Areas     : 30
...
  Columns   : ['group', 'mean', 'sd', 'ci_lower', 'ci_upper', 'rse_pct', 'mse', 'rmse']
...
```

The per-area table is `result_table`; `mean_rse` and `mean_mse` on the same object give
the averages across areas at a glance.

The R-style alias `hbsae` is the same function. To estimate for areas that were not in
the fitting data, pass `new_data=`. The frame needs only the predictor columns and the
group column: the response and the survey-design columns (`sampling_var`, `n`/`deff`,
`trials`) do not enter the area mean and may be absent. A row whose group label was not
seen during fitting is a non-sampled area. At each posterior draw its area effect is
taken from a randomly chosen fitted area — the approach brms calls
`sample_new_levels = "uncertainty"` — so its interval is wider than a sampled area's, as
it should be: no survey data speaks for it. Those draws are seeded by the model's
`random_seed`, so repeated calls agree.

Change the interval width with `ci_prob`, which defaults to 0.95:

```{testcode} fitted
est = estimate_areas(model, ci_prob=0.90)
```

## Check it worked

One row per area, with these columns:

| Column | Meaning |
| --- | --- |
| `mean`, `sd` | Posterior mean and standard deviation of the area quantity |
| `mse`, `rmse` | Posterior variance and its square root |
| `rse_pct` | Relative standard error, `sd / \|mean\| × 100` |
| `ci_lower`, `ci_upper` | Highest-density interval at `ci_prob` |

Each input row is one area. If a group label repeats, the model was fitted on unit-level
rows: the table is then per observation, not per area, and a warning is logged.

The check that matters is **shrinkage**. Compare `mean` against the direct estimate in
your data: model-based estimates should sit between the direct estimate and the pattern
predicted by the auxiliary variables, and should move furthest in the areas with the
largest sampling variance. That is the whole point of the exercise. If the estimates
match the direct estimates exactly, the model is contributing nothing and something is
wrong upstream.

`rse_pct` is the usual publication threshold. Areas whose relative standard error is
still large after modelling should be reported with that caveat rather than presented as
equally reliable.

```{note}
These estimates come from the posterior of the area means, not from the posterior
predictive distribution. That is deliberate and is not a tuning choice — see
*Related* below.
```

## If it fails

**The model has not been fitted.** Shown above; the message names the call you skipped.

**`DataValidationError` naming a column.** `new_data` lacks a predictor or the group
column, or every row has a missing value in one of them. Those are the only columns the
area mean needs; add the named column.

**`EstimationError` with a shape in the message.** Something about this model and this
data cannot be estimated. The error carries the family, the shape of `new_data` and the
requested `ci_prob`, which between them usually identify the problem.

**`ValueError` about `ci_prob`.** The interval probability must lie strictly between 0
and 1.

Note that a wrong-order mistake is kept distinguishable from a genuine estimation
failure: calling before `fit()` raises `ModelNotFittedError`, and that is deliberately
*not* rewrapped as `EstimationError`.

**Row counts do not line up with your input.** Rows with missing values are dropped by
the pipeline before estimation, and the area labels are taken from the processed frame
rather than the raw one, so the labels always match the numbers. Compare against
`model.check_data()` rather than against your original DataFrame. For `new_data`, only a
missing predictor or group value drops a row.

## Related

Why the estimate for a small area is pulled towards the model, why that reduces error
rather than introducing bias, and why the posterior predictive distribution would be the
wrong thing to summarise here, are explained in {doc}`../explanation/hb-sae`.
