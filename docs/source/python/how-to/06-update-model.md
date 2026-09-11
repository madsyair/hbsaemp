# 6 · Update a model

Refit an existing specification against new data, a changed formula, changed priors or a
new sampler configuration.

## Before you start

You need a fitted model. `update_model()` is a **full refit**, not a resumption: it
builds and samples again from scratch, keeping the parts of the specification you do not
override. The R package behaves the same way — its `update_hbm()` reruns `brms::brm()`
with fresh chains.

```{testcode}
from hbsaemp import (create_model, load_dataset, update_model,
                     ModelNotFittedError)

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")

try:
    update_model(model)
except ModelNotFittedError as err:
    print(err.message)
```

```{testoutput}
Call GaussianModel.fit() before accessing .result.
```

## Do it

Refit with different sampler settings — the usual reason, after
{doc}`03-check-convergence` says the chains need more:

```python
from hbsaemp import update_model, ModelConfig

updated = update_model(model, config=ModelConfig(draws=4000, tune=2000,
                                                 chains=4, target_accept=0.95))
```

Sampler settings can also be overridden one at a time — `draws`, `tune`, `chains`,
`cores`, `target_accept`, `random_seed`, `max_treedepth`, `progressbar` and
`sampler_kwargs` — on top of the current configuration. Passing both a `config` and an
individual override is refused, because one of them would have to be ignored:

```python
updated = update_model(model, target_accept=0.99, max_treedepth=12)
```

Refit against newer data, keeping the same formula, family and priors:

```python
updated = update_model(model, new_data=df_2025)
```

If the new frame lacks the survey-design columns (`sampling_var`, or `n` and `deff`) but
has the same number of rows, they are copied from the current data with a warning, on the
assumption that the rows are in the same order. With a different number of rows they
cannot be copied and the call raises.

Change the formula with an R-style update template. `.` stands for the current side of
the formula, `+ term` adds a term and `- term` removes one. Without `new_data`, an added
term is read from the current data:

```python
updated = update_model(model, formula=". ~ . + x3 - x1")
```

Change priors by passing only the ones that change. They are merged into the current
priors — new entries win and the rest are kept, as `brms::update()` does — and a prior on
a term the formula update removes is dropped:

```python
from hbsaemp import Prior

updated = update_model(model, priors={"x3": Prior("Normal", mu=0, sigma=1)})
```

The R-style alias `update_hbm` is the same function.

Only what you pass is changed; the family, the link, the grouping and the family-specific
columns are carried over. That is the point of the call — retyping the whole
specification invites a silent difference between the two runs.

## Check it worked

`update_model()` writes the new result, data, config, formula and priors back onto the
model **in place**, so the model always reflects its most recent fit:

```python
print(model.is_fitted)
print(model.result.fitted_at)     # timestamp of the newest run
print(model.config.draws)         # the config actually used
print(model.formula)              # the formula actually fitted
```

This matters when you chain calls. A second `update_model(model, ...)` starts from the
state the first one produced, not from the original construction — so escalating sampler
settings step by step behaves the way you would expect.

Then re-run {doc}`03-check-convergence`. A refit with new settings has new diagnostics,
and the previous verdict does not carry over.

## If it fails

**The model has not been fitted.** Shown above. Update refits an existing fit; there has
to be one.

**`new_data` is not a DataFrame.** The argument is type-checked, so a Series or a dict
is refused immediately with a `TypeError` rather than failing deep inside the pipeline.

**`ValueError` about `config` and overrides.** Pass either a whole `ModelConfig` or
individual settings on top of the current one.

**`FormulaError` from the template.** The template has no `~`, or the result contains a
term that is not a bare column name.

**The new data no longer satisfies the family.** The same validation as a fresh
`create_model()` applies, so a column that has gone missing, turned non-numeric, or
strayed outside the family's domain raises `DataValidationError`. Run
`model.check_data()` on the new frame first if you want to see that before paying for
sampling.

**The refit converges worse than the original.** Nothing is wrong with the call. New
data can be harder to fit than old data — check whether some areas now have far smaller
sample sizes, which shows up as larger sampling variances and slower mixing.

## Related

Why a refit re-enters the workflow at the diagnostics stage rather than continuing from
where the previous fit ended is explained in {doc}`../explanation/workflow-order`.
