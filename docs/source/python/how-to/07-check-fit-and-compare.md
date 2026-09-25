---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 7 · Check the fit and compare models

Check that a model reproduces the data, then rank the candidate models.

## Before you start

You need models that passed {doc}`05-check-convergence`. Compared models must be fitted to
the same response and the same rows; otherwise `compare_models()` raises `ValueError`.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from hbsaemp import compare_models, create_model, load_dataset, ModelConfig

df = load_dataset("data_fhnorm")
config = ModelConfig(target_accept=0.95, progressbar=False)
m1 = create_model("y ~ x1", family="gaussian", data=df,
                  group="group", sampling_var="D", config=config)
m2 = create_model("y ~ x1 + x2", family="gaussian", data=df,
                  group="group", sampling_var="D", config=config)
m1.fit()
m2.fit()
```

## Do it

### Posterior predictive check

```{code-cell} ipython3
check = compare_models(m2)
plt.close(check.params_plot)
```

The plot shows data replicated from the posterior next to the observed data. It is drawn
for one model: the model passed, or the first model of a list. `n_draws_ppc=` sets the
number of replicated datasets (default 100).

### Compare models

```{code-cell} ipython3
cmp = compare_models([m1, m2])
plt.close(cmp.params_plot)
plt.close(cmp.pp_check_plot)
print(cmp.summary())
cmp.comparison_table
```

The R-style alias `hbmc` is the same function.

`cmp.loo`, `cmp.bayes_factor` and `cmp.prior_sensitivity` follow the shape of the input:

| Call | Returns |
| --- | --- |
| `compare_models(model)` | a single result |
| `compare_models([model])` | a dict with the key `"model_0"` |
| `compare_models([m1, m2])` | a dict with one entry per model |

### Test the coefficients

`metrics=["loo", "bf"]` adds a Bayes factor for each coefficient:

```{code-cell} ipython3
tested = compare_models(m2, metrics=["loo", "bf"])
plt.close(tested.params_plot)
plt.close(tested.pp_check_plot)
tested.bayes_factor
```

## Check it worked

**Posterior predictive check.** The model is adequate when the observed data lies within the
replicated data. A systematic gap, such as a shifted centre, a wrong spread or a missing
tail, means the model does not reproduce the data.

**ELPD** (PSIS-LOO-CV). Higher is better. When the difference between two models
(`elpd_diff`) is smaller than about twice its standard error (`dse`), the data cannot tell
them apart.

**Pareto k.** Observations with *k* above about 0.7 make the LOO estimate unreliable.
Fay-Herriot models often have some, because every area has one observation and its own
random effect. When many are flagged, treat the ranking as indicative only.

**Bayes factor.** `BF10` is the evidence that a coefficient differs from zero (Savage-Dickey).
It tests one coefficient within one model, unlike R's `hbmc()`, which compares whole models
by bridge sampling. A vague prior on the coefficient inflates the evidence for zero.

- **Adequate:** continue to {doc}`08-prior-sensitivity`.
- **Not adequate:** go back to {doc}`02-specify-model` and change the family, the auxiliary
  variables or the link.

## If it fails

**WAIC is requested.** It is not available; use LOO.

```{code-cell} ipython3
try:
    compare_models(m1, metrics=["waic"])
except NotImplementedError as err:
    print(type(err).__name__)
```

**`ValueError` before anything is computed.** The list of models is empty, `metrics` is
empty, `n_draws_ppc` is not a positive integer, or the models were fitted to different data.

**`ModelNotFittedError`.** Fit every model first.

**The log-likelihood group is missing.** LOO needs the log-likelihood that `fit()` computes.
A model fitted by other means raises `ValueError`.

**A plot is missing.** The reason is stored in `cmp.plot_errors`. The table is still
returned.

## Related

Why comparison comes after convergence and before estimation:
{doc}`../explanation/workflow-order`.
