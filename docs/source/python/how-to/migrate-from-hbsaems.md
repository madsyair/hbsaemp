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

```python
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

## Related

Why the Python package offers three ways to build the same model, when R offers one, and
which one corresponds most closely to an `hbsaems` call, is explained in
{doc}`../explanation/three-tier-api`.
