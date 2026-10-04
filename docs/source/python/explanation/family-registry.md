# Why family metadata lives in one registry

Every fact about a family is written once, so the interfaces, the data checks and the fitted
model always agree.

## The question

`family="beta"` decides several things at once: which extra arguments are valid, which links
are allowed, the response domain, which column the preprocessor derives, which parameter the
survey design fixes, and the name of the mean parameter in the posterior.

These facts are needed at different moments, by code in different modules. The obvious design
stores each fact where it is used. This package does not.

## Background

hbsaems keeps a model registry. Each family is a spec holding its brms family, its link, a
response check and a template for the formula's left-hand side, and both `hbm()` and
`hbm_flex()` look families up in it (Choir et al., 2026).

hbsaemp keeps the same idea in `FAMILY_SPECS`: one frozen `FamilySpec` per family, read
through `list_families()` and `get_family_spec()`. A spec holds four kinds of fact:

| Kind | Fields | For `beta` |
| --- | --- | --- |
| Backend | `bambi_family` | the Bambi family `beta` |
| Metadata | `mean_param_key`, `default_link`, `supported_links` | mean `mu`; link `logit`, or `probit` |
| Wiring | `pipeline_fields`, `user_params`, `required_params` | the arguments `n`, `deff`, `squeeze` |
| Behaviour | `response_check`, `preprocess`, `addition_template`, `fixed_params` | response in $(0, 1)$; adds `log_phi`; `kappa` fixed from `n` and `deff` |

The three families of this version compare as follows:

| Family | Response | Default link | Family columns | Fixed by the design |
| --- | --- | --- | --- | --- |
| `gaussian` | any real value | `identity` | `sampling_var` | `sigma`, at $\sqrt{\psi_i}$ |
| `beta` | a proportion in $(0, 1)$ | `logit` | `n`, `deff`, `squeeze` | `kappa`, at $n_i / \text{deff}_i - 1$ |
| `binomial` | a count, at most `trials` | `logit` | `trials` (required) | nothing |

## The reasoning

### The risk is silent disagreement

If the constructor's list of valid arguments lived in one file and the preprocessor's list of
derived columns in another, updating one without the other would give a package that accepts
an argument and then ignores it.

Nothing would raise. The user passes `deff="deff"`, the constructor accepts it, the precision
is never derived, and a different model is fitted from the one requested. The estimates would
still look plausible, which is how such a mistake survives. With one registry, every reader
consults the same entry.

### One source, several readers

| Reader | What it reads | When |
| --- | --- | --- |
| `create_model()` | valid and required arguments, parameters that can be fixed | at construction |
| `DataValidator` | the response domain check | in `check_data()`, `check_prior()` and `fit()` |
| `DataPreprocessor` | the offset transform, such as `log_phi` | in `check_data()`, `check_prior()` and `fit()` |
| the model | the link check, the formula's left-hand side, the fixed parameters, the Bambi family | in `check_prior()` and `fit()` |
| `predict()`, `estimate_areas()` | the mean parameter, `mu` or `p` | after `fit()` |
| `update_model()` | the design columns to carry over to new data | at a refit |

The test of the design is that **adding a family means adding an entry**, plus the behaviour
specific to that family, rather than editing every reader.

### A fixed parameter is a fact, not code

The registry states that the Beta precision and the Gaussian scale are computed from the survey
design, as a `FixedParam`. The model turns that statement into Bambi terms: a sub-formula for
the parameter with the design column as an offset, and an intercept held at zero by a normal
prior with standard deviation 0.001. A caller's `fixed_params=` becomes the same kind of entry,
so one mechanism serves both.

The fix is tight rather than exact. hbsaems writes `0 + offset(...)`, which Bambi cannot
predict from, so the intercept prior holds the parameter within about 0.1% of its value. The
intercept shows in the diagnostics as `sigma_Intercept` or `kappa_Intercept`.

### Why arguments from another family raise

The registry knows which arguments belong to `gaussian`, so it knows `trials` does not.
`create_model(..., family="gaussian", trials="n")` raises a `ValueError` listing the valid
arguments, instead of dropping `trials` and fitting a Gaussian model to a binomial question.
The same entry sets the arity: `n` and `deff` together fix `kappa`, so one without the other
raises too.

### Why the registry is read-only at run time

`list_families()` and `get_family_spec()` are public, and each `FamilySpec` is frozen. hbsaems
lets a user add a family at run time with `register_hbsae_model()`. In hbsaemp a family is a
change to the package: it needs its `FamilySpec` and a model class in `MODEL_REGISTRY`, and a
spec whose parts do not fit together is refused when the package is imported. The two
registries change together, in a reviewed change to the code, instead of separately in a
user's session.

### Why a planned family gets its own message

The package can mark a name as known but not yet implemented. `family="lognormal"` raises a
message saying it is planned for version 2, not the error for a typo. A typo is fixed by
correcting the spelling; a planned family by waiting or choosing another.

## Consequences

**Adding a family is local.** It takes a registry entry and a model class. The three tiers
stay as they are, since `hbm_flex` reaches a new family through `aux_args` and `addition_var`.
A tier 1 function for it, such as an `hbm_poisson`, is a convenience added on top.

**Errors name the alternatives.** The valid families and arguments are known at the moment of
failure, so the message says what would work.

**Validation runs before the backend loads.** The registry is plain Python data, so family,
link and argument checks finish in milliseconds, before Bambi is imported.

**The registry can be inspected.** `get_family_spec("beta")` shows the default link and the
parameter that can be fixed, as the case-study tutorial does before it builds its model.

**A new family is small but not free.** The entry is short; the behaviour it promises, the
domain check, the transformations and the formula shape, is the real work.

## Related

How to choose a family and supply its columns: {doc}`../how-to/02-specify-model`.

## References

- Choir, A. S., Nurhayati, S. S., Zamzanah, S., & Oktalia Siregar, A. L. (2026). *hbsaems:
  Hierarchical Bayesian Area-Level Small Area Estimation Models*. R package version 1.1.0.
