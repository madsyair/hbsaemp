---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 6 · Update a model

Refit a model with new sampler settings, new data, a changed formula or changed priors.

## Before you start

You need a fitted model. `update_model()` refits from scratch and keeps every part of the
specification you do not change, as R's `update_hbm()` does.

```{code-cell} ipython3
from hbsaemp import create_model, load_dataset, update_model, ModelConfig

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                     sampling_var="D", config=ModelConfig(progressbar=False))
model.fit()
```

## Do it

When {doc}`05-check-convergence` fails, refit with stronger sampler settings:

```{code-cell} ipython3
updated = update_model(model, config=ModelConfig(draws=2000, tune=2000, chains=4,
                                                 target_accept=0.95, progressbar=False))
```

Or change single settings on top of the current configuration: `draws`, `tune`, `chains`,
`cores`, `target_accept`, `random_seed`, `max_treedepth`, `progressbar` or
`sampler_kwargs`. Passing `config=` together with a single setting raises `ValueError`.

```{code-cell} ipython3
updated = update_model(model, target_accept=0.99, max_treedepth=12)
```

The same call refits on new data, keeping the formula, family and priors:

```{code-cell} ipython3
df_2025 = load_dataset("data_fhnorm")  # stands in for a newer survey round
updated = update_model(model, new_data=df_2025)
```

If `new_data` lacks a survey-design column (`sampling_var`, `n`, `deff`, or a column named in
`fixed_params`) but has the same number of rows, the column is copied from the current data
with a warning. The rows are assumed to be in the same order. With a different number of
rows, the call raises.

Change the formula with an R-style template. `.` keeps the current side, `+ term` adds a
term and `- term` removes one. An added term is read from the current data unless
`new_data` is given.

```{code-cell} ipython3
updated = update_model(model, formula=". ~ . + x3 - x1")
```

New priors are merged into the current ones, and a prior on a removed term is dropped:

```{code-cell} ipython3
from hbsaemp import Prior

updated = update_model(model, priors={"x3": Prior("Normal", mu=0, sigma=1)})
```

The R-style alias `update_hbm` is the same function.

## Check it worked

`update_model()` updates the model in place:

```{code-cell} ipython3
print(model.is_fitted)
print(model.result.fitted_at)     # time of the latest fit
print(model.config.draws)         # config of the latest fit
print(model.formula)              # formula of the latest fit
```

A second `update_model(model, ...)` starts from this state, so you can raise the settings
step by step.

Then return to {doc}`05-check-convergence`. The earlier verdict does not apply to the new
fit.

## If it fails

**`ModelNotFittedError`.** Fit the model first.

**`TypeError`.** `new_data` must be a DataFrame.

**`ValueError` about `config`.** Pass a whole `ModelConfig` or single settings, not both.

**`FormulaError`.** The template has no `~`, or a term is not a column name.

**`DataValidationError`.** The new data fails the family's checks. Build a model on the new
data and run `check_data()` to see the problem before sampling.

**The refit converges worse than before.** New data can be harder to fit, for example when
some areas now have much smaller samples.

## Related

Why a refit returns to the convergence check: {doc}`../explanation/workflow-order`.
