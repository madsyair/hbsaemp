# Why there are three ways to build the same model

From the advanced tier down to the beginner tier, each asks for one decision fewer, and all
three end in the same model.

## The question

Three functions build the same model. `hbm_gaussian()` takes a response and a list of
auxiliary variables. `hbm_flex()` takes the same plus a family. `create_model()` takes a
formula. Which one is the real API, and what are the others for?

## Background

The R package hbsaems arranges its modelling functions in three layers (Choir et al., 2026):
SAE-friendly wrappers such as `hbm_betalogitnorm()`, the flexible `hbm_flex()`, and the
universal `hbm()`. Most users start with a wrapper and step up a layer when a model needs
more control.

hbsaemp keeps that design and names each tier after the programming it expects from the
user:

| Tier | User | Function | You supply | Decided for you |
| --- | --- | --- | --- | --- |
| 1 | Beginner | `hbm_gaussian`, `hbm_beta`, `hbm_binomial` | response, auxiliary list, the family's own columns | the family and the formula |
| 2 | Intermediate | `hbm_flex` | response, auxiliary list, family | the formula |
| 3 | Advanced | `create_model` (R alias `hbm`) | a formula | nothing |

The formula at tier 3 is the R-style notation that Bambi reads, as in `y ~ x1 + x2`, the
notation of lme4 and brms (Capretto et al., 2022). The tiers differ only in how much of the
model the user writes.

## The reasoning

### Each tier delegates to the next

The three tiers form one chain, each link removing a decision:

```
hbm_gaussian()  →  hbm_flex()  →  create_model()  →  the model object
```

A tier 1 function fixes the family and calls `hbm_flex`, which writes the formula and calls
`create_model`, where validation, the area random effect and construction happen.

Three separate implementations would be three places for behaviour to drift. A validation
rule added to `create_model` but not to `hbm_beta` would make the same model behave
differently depending on the function used. With one chain, the same specification gives the
same formula, family and columns at every tier, and the test suite holds the three to it.

### Where a family's own arguments live

Only tier 1 names a family's columns: `hbm_beta` takes `n`, `deff` and `squeeze`, and
`hbm_binomial` takes `trials`. `hbm_flex` stays family-neutral and passes them through two
generic arguments, as `hbm_flex()` in hbsaems does:

```python
hbm_binomial("y", ["x1"], df, trials="n")                        # tier 1
hbm_flex("y", ["x1"], df, family="binomial", addition_var="n")   # the same model
hbm_beta("y", ["x1"], df, n="n", deff="deff")                    # tier 1
hbm_flex("y", ["x1"], df, family="beta",
         aux_args={"n": "n", "deff": "deff"})                    # the same model
```

`addition_var` is the column the likelihood needs beside the response, and `aux_args` holds
the rest. `sampling_var` is the one family column that `hbm_flex` names directly, as hbsaems
keeps `sampling_variance` at that layer. A new family declares its columns in the registry
({doc}`family-registry`) and reaches `hbm_flex` through these channels, so `hbm_flex` never
changes.

### Why the signatures are narrow

No interface accepts a catch-all `**kwargs`. Every argument is named, and each tier 1 function
exposes only its own family's arguments. `hbm_gaussian(..., trials="n")` is therefore a
`TypeError` raised by Python at the call, naming the argument.

The same rule decides `fixed_params=`. `hbm_gaussian` and `hbm_beta` take it because their
family has a parameter that can be fixed; `hbm_binomial` does not, since its only parameter
is the one being estimated. A permissive signature would pass a mistake several layers down,
or ignore it and build a different model from the one requested.

### Why the advanced tier still exists

Tiers 1 and 2 add one random-effect term, the random intercept `(1|area)`, through
`area_var=`. Any other structure, such as a random slope `(x1|area)`, needs a formula, and a
formula needs `create_model`. A script translated from hbsaems lands here too, since `hbm()`
is `create_model()`.

The formula still has limits. Every fixed-effect term must be a column name, so
transformations and interactions are computed as columns first, at every tier, and the data
checks see every column the model uses. The universal `hbm()` of hbsaems reaches all of brms,
including spatial random effects; the advanced tier here reaches the Bambi formula, and
spatial random effects are planned for a later version.

### One model underneath

Whatever the tier, the result is the same kind of object. One pipeline builds and fits it:
the formula goes to Bambi, PyMC samples it with NUTS, and ArviZ holds the draws. The
functions used after fitting, `check_convergence`, `compare_models`, `estimate_areas` and
`update_model`, never need to know which tier built the model.

## Consequences

**Start with the simplest interface that can express your model.** It has the fewest
arguments to get wrong. Move up a tier when you need something it cannot express.

**Moving between tiers does not change the model.** The same specification at any tier gives
the same formula and the same columns, so the analysis can start at tier 1 and move to tier 3
without starting over.

**`model.formula` is the formula that is fitted.** Whatever the tier, it shows the resolved
formula, including the group term added for you.

**An R script maps onto a tier.** A call to `hbm()` carries over to `create_model()` or its
alias `hbm`. A call to `hbm_betalogitnorm()` or `hbm_binlogitnorm()` maps onto `hbm_beta` or
`hbm_binomial`.

**The graphical interface builds through tier 1.** Its family choice calls `hbm_gaussian`,
`hbm_beta` or `hbm_binomial`, so a model built by clicking is the model a beginner builds in
code.

## Related

How to specify a model with each interface: {doc}`../how-to/02-specify-model`.

## References

- Capretto, T. et al. (2022). Bambi: A simple interface for fitting Bayesian linear models
  in Python. *Journal of Statistical Software*, 103(15).
- Choir, A. S., Nurhayati, S. S., Zamzanah, S., & Oktalia Siregar, A. L. (2026). *hbsaems:
  Hierarchical Bayesian Area-Level Small Area Estimation Models*. R package version 1.1.0.
