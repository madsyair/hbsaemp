# 2 · Fit the model

Run the sampler, and choose how many draws, chains and warm-up steps to spend.

## Before you start

You need a model object from {doc}`01-specify-model`. Fitting is the one expensive step
in the workflow: everything before it takes milliseconds, and this takes minutes. Two
things are worth doing first, because both are free.

Print the model. Before fitting, `summary()` describes what will be sampled rather than
raising, so it is always safe to look:

```{testcode}
from hbsaemp import create_model, load_dataset

df = load_dataset("data_fhnorm")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")
print(model.summary().splitlines()[0])
```

```{testoutput}
GaussianModel [not fitted]
```

Then run the data pipeline on its own, so a bad column costs you a millisecond instead
of a failed sampling run:

```{testcode}
print(model.check_data().shape)
```

```{testoutput}
(30, 10)
```

## Do it

Sampler settings are bundled into a `ModelConfig` rather than passed as loose keyword
arguments, so the same settings can be reused across models in a comparison study:

```{testcode}
from hbsaemp import ModelConfig, DEFAULT_CONFIG

print(DEFAULT_CONFIG.draws, DEFAULT_CONFIG.tune,
      DEFAULT_CONFIG.chains, DEFAULT_CONFIG.target_accept)

config = ModelConfig(draws=2000, tune=1000, chains=4, target_accept=0.95)
print(config.draws, config.target_accept)
```

```{testoutput}
1000 1000 4 0.8
2000 0.95
```

Pass it when you build the model, then sample:

```{testcode}
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D", config=config)
result = model.fit()
```

`fit()` returns a `ModelResult` and also stores it on the model, so `model.result` and
the return value are the same object.

### How much to spend

Start at the default and escalate only when the diagnostics tell you to.

| Model | Suggested settings |
| --- | --- |
| Simple area-level model | `ModelConfig()` — the default |
| Fay-Herriot offset, one grouping | `draws=2000, tune=1000, chains=4, target_accept=0.95` |
| Nested groups, many hyperparameters | `draws=4000, tune=2000, chains=4, target_accept=0.95` |

When the sampler struggles, raise `target_accept` first, then `tune`, then `draws`, in
that order. Raising `draws` first is the common mistake: it buys more of the same
badly-explored posterior rather than fixing the exploration.

Keep `chains` at 4, or at the very least 2. R-hat compares chains against each other, so
a single chain makes the most useful convergence diagnostic impossible to compute.

## Check it worked

```{testcode}
print(model.is_fitted)        # True
print(model.result.summary())
```

```{testoutput}
:hide:
:options: +ELLIPSIS

True
ModelResult [gaussian]
  Formula : y ~ x1 + x2 + (1|group)
  n       : 30
...
```

Sampling finishing is not the same as sampling succeeding. The real check is
{doc}`03-check-convergence`, and it is not optional.

## If it fails

**A setting is out of range.** `ModelConfig` validates at construction, so the mistake
surfaces before any sampling time is spent:

```{testcode}
for bad in [{"draws": -1}, {"chains": 0}, {"target_accept": 1.5}]:
    try:
        ModelConfig(**bad)
    except ValueError as err:
        print(type(err).__name__)
```

```{testoutput}
ValueError
ValueError
ValueError
```

**The data is rejected.** Every message names the offending column; see the failure
section of {doc}`01-specify-model`, and reach for `check_data()` to see it sooner.

**Sampling is very slow, or the log mentions divergences.** That is a modelling problem,
not a configuration problem. Divergences mean the sampler could not explore parts of the
posterior; more draws will not fix them. Raise `target_accept` and re-read
{doc}`03-check-convergence`.

**The log shows `ERROR pymc.stats.convergence`.** This comes from PyMC itself when the
effective sample size per chain is small. It is informational and should not be
silenced — treat it as a signal to raise the sampler settings.

## Related

Why fitting sits where it does in the sequence, and why the checks after it cannot be
reordered or skipped, is explained in {doc}`../explanation/workflow-order`.
