# 3 · Check convergence

Decide whether the chains have converged, and what to change when they have not.

## Before you start

You need a fitted model from {doc}`02-fit-model`. Calling this on an unfitted model
raises rather than guessing:

```{testcode}
from hbsaemp import create_model, load_dataset, check_convergence, ModelNotFittedError

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")

try:
    check_convergence(model)
except ModelNotFittedError as err:
    print(err.message)
```

```{testoutput}
Call GaussianModel.fit() before accessing .result.
```

MCMC does not converge or fail; it produces draws whose trustworthiness you have to
assess. This page is that assessment, and skipping it means reporting estimates you
cannot defend.

## Do it

```python
from hbsaemp import check_convergence

diag = check_convergence(model)
print(diag.summary())
```

The R-style alias `hbcc` is the same function.

By default this computes the diagnostics and renders density, rank and energy plots. The
pair plot is opt-in because it is slow:

```python
diag = check_convergence(model, plot_types=["dens", "rhat", "energy", "pair"])
```

## Check it worked

Three numbers decide the verdict. Any one of them out of range triggers a
`ConvergenceWarning`.

| Diagnostic | Acceptable | What it means when it is not |
| --- | --- | --- |
| R-hat | ≤ 1.01 | Chains disagree; they explored different regions |
| Bulk ESS | ≥ 400 | Too few effective draws for the posterior centre |
| Tail ESS | ≥ 400 | Credible intervals are unreliable, even if the mean looks fine |

Read all three. A good R-hat with a poor tail ESS is a common and dangerous
combination: the point estimate looks settled while the interval around it is noise.

`ConvergenceWarning` is a warning, not an error, which has a practical consequence:

```{testcode}
from hbsaemp import ConvergenceWarning, HBSAEError

print(issubclass(ConvergenceWarning, HBSAEError))
print(issubclass(ConvergenceWarning, UserWarning))
```

```{testoutput}
False
True
```

`except HBSAEError` will not catch it. Sampling completed; the result is merely
untrustworthy. To handle it programmatically, use `warnings.catch_warnings`.

## If it fails

Escalate in this order, and re-fit after each change:

1. **Raise `target_accept`** to 0.95, then 0.99. This makes the sampler take smaller,
   more careful steps and is the fix for divergences.
2. **Raise `tune`.** More warm-up gives the sampler more time to adapt before the draws
   that count begin.
3. **Raise `draws`.** Only after the first two. More draws from a badly-explored
   posterior is more of the same problem.

If none of that works, the model is the problem rather than the sampler. Too many
hyperparameters for 30 areas, a group with a single observation, or an auxiliary
variable that is nearly collinear with another will all show up as stubborn
non-convergence.

**A plot failed but the diagnostics returned.** Plot failures are recorded in
`diag.plot_errors` rather than being raised, so a broken plot never costs you the
numbers. Check that list if a figure you expected is missing.

**PyMC logs `ERROR pymc.stats.convergence`.** That is PyMC reporting low per-chain
effective sample size. It is informational, and silencing it would only hide the signal
you are looking for.

## Related

Why this check comes after fitting and before any estimate is read, and why that order
carries statistical meaning rather than being a convention, is explained in
{doc}`../explanation/workflow-order`.
