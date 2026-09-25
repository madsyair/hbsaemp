---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
  pygments_lexer: ipython3
---

# 8 · Check prior sensitivity

Measure how much each posterior depends on its prior.

## Before you start

You need the model chosen in {doc}`07-check-fit-and-compare`:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from hbsaemp import compare_models, create_model, load_dataset, ModelConfig

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                     sampling_var="D", config=ModelConfig(target_accept=0.95,
                                                          progressbar=False))
model.fit()
```

## Do it

```{code-cell} ipython3
cmp = compare_models(model, run_prior_sensitivity=True,
                     sensitivity_vars=["Intercept", "x1", "x2", "1|group_sigma"])
plt.close(cmp.params_plot)
plt.close(cmp.pp_check_plot)
cmp.prior_sensitivity
```

The check scales the prior and the likelihood by a small power and measures how much each
posterior changes. It is the method of R's priorsense package. Without `sensitivity_vars=`,
it covers every scalar and group-level parameter.

## Check it worked

| Column | Meaning |
| --- | --- |
| `prior` | Sensitivity to scaling the prior |
| `likelihood` | Sensitivity to scaling the likelihood |
| `diagnosis` | The verdict for the parameter |

A value above 0.05 counts as sensitive. The `diagnosis` column flags two cases:

| Diagnosis | Meaning | What to do |
| --- | --- | --- |
| potential prior-data conflict | Prior and likelihood are both sensitive: they disagree | Revise the prior in {doc}`02-specify-model` and check it again in {doc}`03-check-priors` |
| potential strong prior / weak likelihood | Only the prior is sensitive: the data says little about the parameter | Report the prior with the result, or choose a prior you can justify |

With no flagged parameter, continue to {doc}`09-estimate-areas`.

## If it fails

**`ValueError`.** `sensitivity_vars=` is given without `run_prior_sensitivity=True`:

```{code-cell} ipython3
try:
    compare_models(model, sensitivity_vars=["x1"])
except ValueError as err:
    print(err)
```

**`ModelNotFittedError`.** Fit the model first.

## Related

Why the prior is checked both before and after fitting: {doc}`../explanation/workflow-order`.
