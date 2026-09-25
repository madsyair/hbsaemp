# hbsaemp

**Hierarchical Bayesian Small Area Estimation** — Python port of R package
[`hbsaems`](https://github.com/madsyair/hbsaems) (Choir et al., 2025).

Design philosophy: **one function for all models**, like R's `caret::train()`.

📖 **Documentation: <https://madsyair.github.io/hbsaemp/>**

## Quick start

```python
from hbsaemp import create_model, ModelConfig

# 1. Configure sampler (= trainControl in caret)
cfg = ModelConfig(draws=2000, tune=1000, chains=4, cores=2)

# 2. create_model() selects the model via family= (= method= in caret)
m1 = create_model("y ~ x1 + x2", family="gaussian",  data=df, group="area",  config=cfg)
m2 = create_model("y ~ x1 + x2", family="beta",       data=df, n="n", deff="deff", config=cfg)
m3 = create_model("y ~ x1 + x2", family="binomial",   data=df, trials="n",   config=cfg)

# 3. Fit (v1+: Bambi MCMC backend)
m1.fit()

# 4. Bayesian workflow — same API for every family
check_prior(m1)              # prior predictive check  (alias: hbpc)  — see note below
check_convergence(m1)        # Rhat, ESS, trace plots  (alias: hbcc)
compare_models([m1, m2])     # LOO, pp_check           (alias: hbmc)
estimate_areas(m1)           # RSE, MSE, RMSE, CI      (alias: hbsae)

# Prior draws without fitting — the model stays unfitted
m1.prior_predictive_idata(draws=500)
m1.check_data()              # validate + preprocess, no MCMC, no Bambi import

# R-style alias also works (for users migrating from hbsaems)
from hbsaemp import hbm
m1 = hbm("y ~ x1 + x2", family="gaussian", data=df, config=cfg)  # identical

# Custom priors — validated at construction (fails fast, before fit)
from hbsaemp import Prior
m4 = create_model("y ~ x1 + x2", family="gaussian", data=df, group="area",
                  priors={"Intercept": Prior("Normal", mu=0, sigma=1)}, config=cfg)

# Inspect available families without touching internals
from hbsaemp import list_families, get_family_spec
list_families()                            # ['beta', 'binomial', 'gaussian']
get_family_spec("beta").supported_links    # frozenset({'logit', 'probit'})

# 5. Web GUI (v1+: Panel dashboard)
from hbsaemp import launch_app
launch_app()                 # ≡ run_sae_app() in R hbsaems
```

## Version roadmap

| Version | Description                       | Extra deps                         |
| ------- | --------------------------------- | ---------------------------------- |
| `0.0.0` | Architecture template — all stubs | `numpy`, `pandas` only             |
| `1.0.0` | Bambi implementation              | `pip install "hbsaemp[bambi,gui]"` |
| `2.0.0` | Spatial extension (CAR/ICAR/SAR)  | `pip install "hbsaemp[spatial]"`   |

## Package structure

```
hbsaemp/
├── _logging.py              configure_logging
├── _exceptions.py           HBSAEError hierarchy
├── _types.py                PEP 695 type aliases
├── py.typed                 PEP 561 marker (package ships its type hints)
├── models/
│   ├── _config.py           ModelConfig  (= trainControl)
│   ├── _base.py             BaseModel ABC · ModelResult dataclass
│   ├── _factory.py          create_model() / hbm() (tier 3) · MODEL_REGISTRY
│   ├── _family_spec.py      FamilySpec · FAMILY_SPECS (single source of truth)
│   ├── _flex.py             hbm_flex (tier 2)
│   ├── _shortcuts.py        hbm_{gaussian,beta,binomial} (tier 1)
│   ├── _prior.py            Prior value object
│   └── _gaussian.py · _beta.py · _binomial.py   family hooks
├── data/
│   ├── _validator.py        DataValidator  (domain checks, read-only)
│   ├── _preprocessor.py     DataPreprocessor  (offset transforms)
│   └── datasets.py          load_dataset() · AVAILABLE_DATASETS
├── diagnostics/
│   ├── convergence.py       check_convergence() / hbcc
│   ├── prior_check.py       check_prior()       / hbpc
│   └── comparison.py        compare_models()    / hbmc
├── estimation/
│   ├── areas.py             estimate_areas()    / hbsae
│   └── update.py            update_model()      / update_hbm
├── utils/
│   └── _formula.py          parse_formula()
└── app/
    ├── _config.py           AppConfig
    ├── _app.py              App stub → v1 (Panel dashboard)
    ├── __init__.py          launch_app() (= run_sae_app())
    └── tabs/                4 tab stubs → v1
```

## License

GPL-3.0-or-later. Based on [hbsaems](https://github.com/madsyair/hbsaems) by Choir et al.

## References

- Choir, A.S. et al. (2025). _hbsaems: Hierarchical Bayesian SAE Models_. R package.
- Capretto, T. et al. (2022). [Bambi](https://doi.org/10.18637/jss.v103.i15). _JSS_ 103(15).
- Rao, J.N.K. & Molina, I. (2015). _Small Area Estimation_ (2nd ed.). Wiley.
