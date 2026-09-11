# 4 · Compare models

Rank competing specifications, and read the comparison table without over-reading it.

## Before you start

You need two or more **fitted** models built from the same response and the same rows.
Comparing models fitted to different data answers nothing: the scores are sums over
observations, so a model fitted to fewer rows will look better for no good reason.
`compare_models()` checks this and raises `ValueError` when the observed responses differ,
even if the row counts happen to match.

Each model must have passed {doc}`03-check-convergence` first. Comparing two sets of
untrustworthy draws produces a trustworthy-looking ranking of nothing.

## Do it

```python
from hbsaemp import compare_models

m1 = create_model("y ~ x1", family="gaussian", data=df,
                  group="group", sampling_var="D")
m2 = create_model("y ~ x1 + x2", family="gaussian", data=df,
                  group="group", sampling_var="D")
m1.fit()
m2.fit()

cmp = compare_models([m1, m2])
print(cmp.summary())
```

The R-style alias `hbmc` is the same function.

Two optional analyses run per model on request:

```python
cmp = compare_models([m1, m2], metrics=["loo", "bf"], run_prior_sensitivity=True)
cmp.bayes_factor["model_1"]        # BF10 / BF01 per coefficient
cmp.prior_sensitivity["model_1"]   # prior / likelihood sensitivity per parameter
```

The shape of the output follows the shape of the input, which matters when you write
code around it. The rule is the same for `loo`, `bayes_factor` and `prior_sensitivity`:

| Call | Returns |
| --- | --- |
| `compare_models(model)` | a single result |
| `compare_models([model])` | a dict keyed `"model_0"` |
| `compare_models([m1, m2])` | a dict, one entry per model |

Passing a list always gives a dict, even a list of one.

## Check it worked

The comparison is based on **PSIS-LOO-CV**, reported as ELPD — expected log pointwise
predictive density. Higher is better.

Read the standard error next to it before declaring a winner. A difference in ELPD
smaller than roughly twice its standard error is not evidence that one model is better;
it is evidence that this data cannot tell them apart. With 30 areas that happens often,
and reporting it honestly is more useful than picking the larger number.

Check the **Pareto k** line of the summary as well. Each observation whose *k* exceeds
the threshold (about 0.7) makes the LOO estimate unreliable, and the comparison with it.
Fay-Herriot models are prone to this: every area has one observation and its own random
effect, so leaving that observation out moves the posterior a lot. When many
observations are flagged, treat the ELPD ranking as indicative only.

**Bayes factors.** `metrics=["bf"]` adds a Savage-Dickey Bayes factor for every
fixed-effect coefficient: `BF10` is the evidence that the coefficient differs from zero,
against it being zero. It is a test of one coefficient within one model, not the
model-versus-model Bayes factor of R's `hbmc()`, which uses bridge sampling. It is only
as meaningful as the coefficient's prior: a vague prior inflates the evidence for zero.

**Prior sensitivity.** `run_prior_sensitivity=True` power-scales the prior and the
likelihood — the method of R's priorsense — and reports how strongly each parameter's
posterior reacts. Values above 0.05 count as sensitive; the `diagnosis` column flags a
potential *prior-data conflict* (both sensitive) or a potential *strong prior / weak
likelihood* (only the prior). Restrict the parameters with `sensitivity_vars=[...]`.

WAIC is not available. It was dropped upstream in favour of PSIS-LOO-CV, and asking for
it is refused rather than silently substituted:

```{testcode}
from hbsaemp import create_model, load_dataset, compare_models

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1", family="gaussian", data=df,
                     group="group", sampling_var="D")

try:
    compare_models(model, metrics=["waic"])
except NotImplementedError as err:
    print(type(err).__name__)
```

```{testoutput}
NotImplementedError
```

## If it fails

**An argument is rejected before anything is computed.** Arguments that cannot be
honoured raise rather than being quietly ignored, so you never receive a result that
silently disregarded what you asked for:

```{testcode}
try:
    compare_models([])
except ValueError as err:
    print(err.args[0].split(";")[0])

try:
    compare_models(model, n_draws_ppc=0)
except ValueError as err:
    print(type(err).__name__)
```

```{testoutput}
compare_models() requires at least one fitted model
ValueError
```

**A model has not been fitted.** Same guard as everywhere else:

```{testcode}
from hbsaemp import ModelNotFittedError

try:
    compare_models(model)
except ModelNotFittedError as err:
    print(err.message)
```

```{testoutput}
Call GaussianModel.fit() before accessing .result.
```

**The log-likelihood group is missing.** LOO needs per-observation log-likelihood, which
`fit()` computes after sampling. If a model was produced some other way, this check
fails with an explicit message rather than an obscure one from the statistics library.

**A plot is missing but the table is there.** Plots are best-effort; the comparison
table is not. A failed plot never removes the numbers, and the reason is recorded in
`cmp.plot_errors`.

## Related

Why comparison comes after convergence and before estimation — and why reversing them
produces a confident ranking of unusable draws — is explained in
{doc}`../explanation/workflow-order`.
