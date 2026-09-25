# Migrate from the R package

Translate an `hbsaems` script into Python, one call at a time.

## Before you start

Every R function has a Python name, and the R name is kept as an alias. An alias is the
same function object, not a wrapper:

```{testcode}
from hbsaemp import (create_model, hbm, check_convergence, hbcc,
                     compare_models, hbmc, estimate_areas, hbsae,
                     update_model, update_hbm, check_prior, hbpc)

print(hbm is create_model)
print(hbcc is check_convergence, hbmc is compare_models)
print(hbsae is estimate_areas, update_hbm is update_model, hbpc is check_prior)
```

```{testoutput}
True
True True
True True True
```

## Do it

| R `hbsaems` | Python `hbsaemp` |
| --- | --- |
| `hbm(...)` | `hbm(...)` or `create_model(...)` |
| `hbpc(...)` | `hbpc(...)` or `check_prior(...)` |
| `hbcc(...)` | `hbcc(...)` or `check_convergence(...)` |
| `hbmc(...)` | `hbmc(...)` or `compare_models(...)` |
| `hbsae(...)` | `hbsae(...)` or `estimate_areas(...)` |
| `update_hbm(...)` | `update_hbm(...)` or `update_model(...)` |

The arguments of `update_hbm()` map onto these names. Both functions refit from scratch, as
`brms::update()` reruns `brm()`:

| R `update_hbm(...)` | Python `update_model(...)` |
| --- | --- |
| `object` | `model` |
| `newdata = df` | `new_data=df` |
| `formula. = . ~ . + x2` | `formula=". ~ . + x2"` |
| `prior = ...` (through `...`) | `priors={...}`, merged into the current priors |
| `iter = 4000, warmup = 2000` | `draws=2000, tune=2000` (`iter` includes warm-up, `draws` does not) |
| `chains`, `cores` | `chains`, `cores` |
| `control = list(adapt_delta = 0.99)` | `target_accept=0.99` |
| `control = list(max_treedepth = 12)` | `max_treedepth=12` |
| `seed = 1` (through `...`) | `random_seed=1` |

R returns a new `hbmfit` and leaves the original unchanged. `update_model()` updates the
model in place and returns its new `ModelResult`.

### Pinned parameters

`fixed_params` keeps its name. `sampling_variance` in R and `sampling_var=` here both mean
`sigma = sqrt(D)`:

| R `hbsaems` | Python `hbsaemp` |
| --- | --- |
| `sampling_variance = "D"` | `sampling_var="D"` |
| `fixed_params = list(sigma = "sd_col")` | `fixed_params={"sigma": "sd_col"}` |
| `fixed_params = list(sigma = 2)` | `fixed_params={"sigma": 2.0}` |
| `fixed_params = list(phi = ...)` | `fixed_params={"kappa": ...}` |
| `fixed_params = list(sigma = c(...))` | add the vector as a column, pass its name |
| `fixed_params = list(phi = ~ I(n/deff - 1))` | `n=`/`deff=`, or pre-compute a column |

Three differences to watch:

**The Beta precision is `kappa`, not `phi`.** brms calls it `phi` and Bambi calls it
`kappa`. The value is the same, `n/deff - 1`.

**A vector or a one-sided formula becomes a column.** R accepts `c(...)` and
`~ I(n/deff - 1)`. Here the value is a column name or a single number: compute it in pandas
and pass the column name.

**The pin is tight, not exact.** R writes `<par> ~ 0 + offset(...)`. Bambi 0.18 crashes on
the empty design matrix, so hbsaemp writes `1 + offset(...)` and holds the intercept with a
`Normal(0, 1e-3)` prior. The parameter lands within about 0.1% of the pinned value.

The dataset names and their column names are the same as in R:

```{testcode}
from hbsaemp import load_dataset, AVAILABLE_DATASETS

print(AVAILABLE_DATASETS)
print(list(load_dataset("data_fhnorm").columns))
```

```{testoutput}
['data_fhnorm', 'data_betalogitnorm', 'data_binlogitnorm', 'data_lnln']
['y', 'D', 'x1', 'x2', 'x3', 'theta_true', 'u', 'group', 'sre']
```

## Check it worked

Fit both versions on the same bundled dataset and compare the posterior summaries. They
should agree relative to their standard errors, not digit for digit: the two packages use
different samplers and seeds.

## If it fails

These differences are deliberate.

**Building and fitting are separate.** In R, `hbm()` fits. Here it returns an unfitted
model and you call `fit()`. This lets you run {doc}`02-specify-model` and
{doc}`03-check-priors` before any sampling.

```{testsetup}
from hbsaemp import load_dataset

df = load_dataset("data_fhnorm")
```

```{testcode}
model = hbm("y ~ x1 + x2", family="gaussian", data=df, group="group")
model.fit()
```

**Sampler settings go in a config object.** Build a `ModelConfig` and pass it as `config=`.
See {doc}`04-fit-model`.

**No transformations inside the formula.** R accepts `log(x)` and `x1:x2`. Here every term
must be a column name, and anything else raises. Compute the column first, as
{doc}`02-specify-model` shows.

**Family-specific arguments are checked.** Passing `trials=` to a Gaussian model raises
instead of being ignored.

**The lognormal family is not available.** `data_lnln` loads, but `family="lognormal"`
raises a roadmap error.

**WAIC is not available.** Model comparison uses PSIS-LOO-CV only. See
{doc}`07-check-fit-and-compare`.

**The Bayes factor answers a different question.** In R, `comparison_metrics = "bf"`
compares two whole models by bridge sampling. Here `metrics=["bf"]` gives a Savage-Dickey
Bayes factor for each coefficient of one model. To weigh two nested models, read the Bayes
factor of the coefficient that separates them.

## Related

Why the Python package has three interfaces where R has one:
{doc}`../explanation/three-tier-api`.
