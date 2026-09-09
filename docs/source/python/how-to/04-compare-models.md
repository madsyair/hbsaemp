# 4 · Compare models

Rank competing specifications, and read the comparison table without over-reading it.

## Before you start

You need two or more **fitted** models built from the same response and the same rows.
Comparing models fitted to different data answers nothing: the scores are sums over
observations, so a model fitted to fewer rows will look better for no good reason.

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

The shape of the output follows the shape of the input, which matters when you write
code around it:

| Call | Returns |
| --- | --- |
| `compare_models(model)` | a single ELPD result |
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
table is not. A failed plot never removes the numbers.

## Related

Why comparison comes after convergence and before estimation — and why reversing them
produces a confident ranking of unusable draws — is explained in
{doc}`../explanation/workflow-order`.
