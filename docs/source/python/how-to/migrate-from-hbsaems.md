# Migrate from the R package

Translate an existing `hbsaems` script into its Python equivalent, one call at a time.

## Before you start

The two packages share a design, so most scripts translate call for call rather than
needing a rewrite. Every R function has a Python name, and the R name is kept as an
alias so a translated script stays recognisable to whoever wrote the original.

The aliases are not wrappers. They are the same function object under a second name,
which means identical arguments, identical behaviour, and nothing to keep in sync:

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

The arguments of `update_hbm()` map onto Python names. Both refit from scratch rather than
continuing the previous chains — `brms::update()` reruns `brm()` too:

| R `update_hbm(...)` | Python `update_model(...)` |
| --- | --- |
| `object` | `model` |
| `newdata = df` | `new_data=df` |
| `formula. = . ~ . + x2` | `formula=". ~ . + x2"` |
| `prior = ...` (through `...`) | `priors={...}` — merged into the current priors, new entries win |
| `iter = 4000, warmup = 2000` | `draws=2000, tune=2000` — `iter` counts warmup, `draws` does not |
| `chains`, `cores` | `chains`, `cores` |
| `control = list(adapt_delta = 0.99)` | `target_accept=0.99` |
| `control = list(max_treedepth = 12)` | `max_treedepth=12` |
| `seed = 1` (through `...`) | `random_seed=1` |

One difference is in the result: R returns a new `hbmfit` and leaves the original
untouched, while `update_model()` updates the model in place and returns its new
`ModelResult`.

### Pinned parameters

R's `fixed_params` carries over under the same name, and `sampling_variance` is the same
convenience over it in both packages — in R it means `fixed_params = list(sigma =
sqrt(D))`, and `sampling_var=` here computes exactly that:

| R `hbsaems` | Python `hbsaemp` |
| --- | --- |
| `sampling_variance = "D"` | `sampling_var="D"` |
| `fixed_params = list(sigma = "sd_col")` | `fixed_params={"sigma": "sd_col"}` |
| `fixed_params = list(sigma = 2)` | `fixed_params={"sigma": 2.0}` |
| `fixed_params = list(phi = ...)` | `fixed_params={"kappa": ...}` |
| `fixed_params = list(sigma = c(...))` | add the vector as a column, pass its name |
| `fixed_params = list(phi = ~ I(n/deff - 1))` | `n=`/`deff=`, or pre-compute a column |

Three things to watch when translating:

**The Beta precision is called `kappa`, not `phi`.** R builds on brms, which names it
`phi`; Python builds on Bambi, which names it `kappa`. It is the same parameter and the
same value, `n/deff - 1`.

**A vector or a one-sided formula becomes a column.** R accepts `c(...)` and `~ I(n/deff
- 1)` directly. Here the value is either a column name or a single number, which is the
same rule that bars transformations inside a formula: compute it in pandas first and pass
the column name. The column then survives into `result.data`, so what was fitted stays
readable.

**The pin is tight, not exact.** R writes `<par> ~ 0 + offset(...)`. Bambi 0.18 crashes on
an empty design matrix there, so hbsaemp writes `1 + offset(...)` and clamps the intercept
with a `Normal(0, 1e-3)` prior instead. The prior outweighs the likelihood by roughly five
orders of magnitude, so the parameter lands within about 0.1% of the value you pinned
rather than exactly on it.

The dataset names carry over unchanged — `data_fhnorm`, `data_betalogitnorm`,
`data_binlogitnorm`, `data_lnln` — and so do the column names inside them:

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

Fit both versions on the same bundled dataset and compare the posterior summaries.
Expect them to agree in substance, not digit for digit: the two packages use different
samplers and different seeds, so the Monte Carlo error differs even when the model is
identical. Point estimates should be close relative to their standard errors.

## If it fails

These are deliberate differences, not bugs.

**Construction and sampling are separate.** In R, `hbm()` fits. Here it returns an
unfitted model and you call `fit()` yourself. That split is what makes
{doc}`01-specify-model` cheap to check, and what allows a prior predictive check before
any sampling time is spent.

```{testsetup}
from hbsaemp import load_dataset

df = load_dataset("data_fhnorm")
```

```{testcode}
model = hbm("y ~ x1 + x2", family="gaussian", data=df, group="group")
model.fit()
```

**Sampler settings live in a config object.** Instead of loose arguments on the fitting
call, build a `ModelConfig` and pass it as `config=`. See {doc}`02-fit-model`.

**No transformations inside the formula.** R formulas accept `log(x)` and `x1:x2`; here
every term must be a bare column name, and anything else is rejected rather than quietly
dropped. Compute the column first — {doc}`01-specify-model` shows how.

**Family-specific arguments are checked against the family.** Passing `trials=` to a
Gaussian model raises instead of being ignored, so a translation slip is caught at once
rather than producing a model that quietly is not the one you meant.

**The lognormal family is not available.** `data_lnln` still loads, but
`family="lognormal"` raises a roadmap error. There is no Python equivalent yet.

**WAIC is gone.** Model comparison uses PSIS-LOO-CV only; asking for WAIC raises. See
{doc}`04-compare-models`.

**The Bayes factor answers a different question.** In R, `comparison_metrics = "bf"`
compares two whole models by bridge sampling. Here `metrics=["bf"]` gives a
Savage-Dickey Bayes factor for each coefficient of one model against zero. To weigh two
nested specifications, read the Bayes factor of the coefficient that separates them.

## Related

Why the Python package offers three ways to build the same model, when R offers one, and
which one corresponds most closely to an `hbsaems` call, is explained in
{doc}`../explanation/three-tier-api`.
