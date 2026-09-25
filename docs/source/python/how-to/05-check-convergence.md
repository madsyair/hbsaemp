---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 5 · Check convergence

Decide whether the chains have converged.

## Before you start

You need a fitted model from {doc}`04-fit-model`:

```{code-cell} ipython3
from hbsaemp import check_convergence, create_model, load_dataset, ModelConfig

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                     sampling_var="D", config=ModelConfig(progressbar=False))
model.fit()
```

## Do it

```{code-cell} ipython3
diag = check_convergence(model)
print(diag.summary())
```

The R-style alias `hbcc` is the same function. The call also draws trace, density,
autocorrelation, R-hat, ESS and energy plots. The pair plot marks divergent transitions; it
is slow, so request it explicitly:

```{code-cell} ipython3
diag = check_convergence(model, plot_types=["dens", "rhat", "energy", "pair"])
```

## Check it worked

A value outside these limits raises a `ConvergenceWarning`.

| Diagnostic | Acceptable | If not |
| --- | --- | --- |
| R-hat | ≤ 1.01 | The chains explored different regions |
| Bulk ESS | ≥ 100 × chains (400 with the default 4) | Too few effective draws for the posterior centre |
| Tail ESS | ≥ 100 × chains | The credible intervals are unreliable |
| Divergences | 0 | Draws near the divergence are biased |
| Tree depth hits | 0 | Trajectories were cut short, so exploration is slow |
| E-BFMI | ≥ 0.3 in every chain | The sampler moves poorly between energy levels |

Check every row. A good R-hat with a poor tail ESS means the point estimate is stable but
its interval is not.

`diag.rhat_ess` holds the values per parameter, `diag.diagnose` the sampler-level counts,
and `diag.ess_threshold` the ESS limit that was applied.

`ConvergenceWarning` is a `UserWarning`, not an `HBSAEError`:

```{code-cell} ipython3
from hbsaemp import ConvergenceWarning, HBSAEError

print(issubclass(ConvergenceWarning, HBSAEError))
print(issubclass(ConvergenceWarning, UserWarning))
```

`except HBSAEError` therefore does not catch it. Use `warnings.catch_warnings` to handle it
in code.

- **Converged:** continue to {doc}`07-check-fit-and-compare`.
- **Not converged:** refit with {doc}`06-update-model`, then check again.

## If it fails

Change one setting at a time, in this order, and refit after each change:

1. **`target_accept`** to 0.95, then 0.99. The sampler takes smaller steps, which removes
   divergences.
2. **`tune`**. More warm-up before the draws that count.
3. **`draws`**. Only after the first two.

If none of these work, the problem is the model: too many hyperparameters for the number of
areas, a group with a single observation, or two nearly collinear auxiliary variables. Go
back to {doc}`02-specify-model`.

**A plot is missing.** The reason is stored in `diag.plot_errors`. The diagnostics are still
returned.

**`ModelNotFittedError`.** Fit the model first.

## Related

Why convergence is checked before any estimate is read: {doc}`../explanation/workflow-order`.
