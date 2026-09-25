---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 4 · Fit the model

Run the sampler and choose the number of draws, chains and tuning steps.

## Before you start

You need a model whose priors passed {doc}`03-check-priors`. Fitting is the slow step: it
takes minutes, while every earlier step takes seconds.

## Do it

Sampler settings go in a `ModelConfig`. The defaults are:

```{code-cell} ipython3
from hbsaemp import create_model, load_dataset, ModelConfig, DEFAULT_CONFIG

print(DEFAULT_CONFIG.draws, DEFAULT_CONFIG.tune,
      DEFAULT_CONFIG.chains, DEFAULT_CONFIG.target_accept)
```

Pass a config when you build the model, then call `fit()`:

```{code-cell} ipython3
df = load_dataset("data_fhnorm")
config = ModelConfig(draws=2000, tune=1000, chains=4, target_accept=0.95,
                     progressbar=False)
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D", config=config)
result = model.fit()
```

`progressbar=False` keeps these pages short. Leave it out to see the sampler's progress.
`fit()` returns a `ModelResult` and also stores it as `model.result`.

### How much to spend

| Model | Settings |
| --- | --- |
| Simple area-level model | `ModelConfig()`, the default |
| Fay-Herriot offset, one grouping | `draws=2000, tune=1000, chains=4, target_accept=0.95` |
| Nested groups, many hyperparameters | `draws=4000, tune=2000, chains=4, target_accept=0.95` |

Use 4 chains, or at least 2. R-hat compares chains, so a single chain cannot be checked.

## Check it worked

```{code-cell} ipython3
print(model.is_fitted)
print(model.result.summary())
```

A finished run is not yet a usable one. Check it in {doc}`05-check-convergence`.

## If it fails

**A setting is out of range.** `ModelConfig` raises `ValueError` as soon as it is created:

```{code-cell} ipython3
for bad in [{"draws": -1}, {"chains": 0}, {"target_accept": 1.5}]:
    try:
        ModelConfig(**bad)
    except ValueError as err:
        print(type(err).__name__)
```

**The data is rejected.** The message names the column. See the errors in
{doc}`02-specify-model`.

**Sampling is slow or reports divergences.** More draws will not fix this. Raise
`target_accept` as described in {doc}`05-check-convergence`.

**The log shows `ERROR pymc.stats.convergence`.** PyMC reports a low effective sample size
per chain. Treat it as a signal to raise the sampler settings; do not silence it.

## Related

Why fitting comes after the prior check: {doc}`../explanation/workflow-order`.
