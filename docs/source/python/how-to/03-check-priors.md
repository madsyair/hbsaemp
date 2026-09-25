---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 3 · Check the priors

Check that the priors produce plausible data before you fit the model.

## Before you start

You need an unfitted model from {doc}`02-specify-model`. The check samples from the priors
only. It takes seconds, and the model stays unfitted.

## Do it

```{code-cell} ipython3
from hbsaemp import check_prior, create_model, load_dataset, ModelConfig

df = load_dataset("data_fhnorm")
config = ModelConfig(random_seed=1, progressbar=False)
model = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                     sampling_var="D", config=config)

prior = check_prior(model)
print(prior.summary())
print(model.is_fitted)
```

The plot shows data simulated from the priors next to the observed data. The R-style alias
`hbpc` is the same function, and `n_draws=` sets the number of simulated datasets
(default 50).

`prior.prior_summary` holds the prior distribution of each parameter:

```{code-cell} ipython3
prior.prior_summary.loc[["Intercept", "x1", "x2", "1|group_sigma"], ["mean", "sd"]]
```

## Check it worked

The priors are reasonable when the simulated data:

- covers the observed data;
- stays on the scale of the observed data, not orders of magnitude wider;
- puts little weight on impossible values, such as a proportion outside 0 to 1.

Compare the middle 95% of the simulated values with the observed range:

```{code-cell} ipython3
import numpy as np

simulated = prior.idata.prior_predictive["y"].values
print(np.quantile(simulated, [0.025, 0.975]).round(1))
print(df["y"].min().round(1), df["y"].max().round(1))
```

The default priors simulate values from about −3 to 12 for data between 3.2 and 6.2. That
is wider than the data but on the same scale, so the priors are reasonable. Continue to
{doc}`04-fit-model`.

## If the priors are not reasonable

Go back to {doc}`02-specify-model`, change `priors=`, and run the check again. For example,
a very wide prior on the coefficients:

```{code-cell} ipython3
from hbsaemp import Prior

vague = {"x1": Prior("Normal", mu=0, sigma=100), "x2": Prior("Normal", mu=0, sigma=100)}
wide = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                    sampling_var="D", config=config, priors=vague)

wide_prior = check_prior(wide)
print(np.quantile(wide_prior.idata.prior_predictive["y"].values, [0.025, 0.975]).round(1))
```

These priors simulate values of about ±200 for data between 3 and 6. Narrow them before
fitting.

## If it fails

**`ImportError`.** The check builds the model in Bambi. Install the `bambi` extra:
`pip install "hbsaemp[bambi] @ git+https://github.com/madsyair/hbsaemp"`.

**`DataValidationError`.** The data fails the same checks as `fit()`. See the errors in
{doc}`02-specify-model`.

**The plot is missing.** `prior.prior_predictive_plot` is `None` and the reason is logged
as a warning. The summary table and `prior.idata` are still returned.

## Related

Why priors are checked before fitting: {doc}`../explanation/workflow-order`.
