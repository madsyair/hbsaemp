# 2 · Specify the model

Choose the family, the auxiliary variables, the area identifier and, if needed, the priors.

## Before you start

You need a DataFrame from {doc}`01-load-data`. Specifying a model does not sample, so every
call on this page returns immediately.

## Do it

Three interfaces build the same model. Use the simplest one that can express yours.

| Interface | Function | You pass | Use it when |
| --- | --- | --- | --- |
| Beginner | `hbm_gaussian`, `hbm_beta`, `hbm_binomial` | response, auxiliary variables, the family's own columns | the response fits one of the three families |
| Intermediate | `hbm_flex` | the same, plus `family=` | the family is chosen in code, for example in a loop |
| Advanced | `create_model` (R alias `hbm`) | a formula | you need a random-effect term other than `(1\|group)`, such as a random slope `(x1\|group)` |

The same Fay-Herriot model, built three ways:

```{testcode}
from hbsaemp import create_model, hbm_flex, hbm_gaussian, load_dataset

df = load_dataset("data_fhnorm")

beginner = hbm_gaussian("y", ["x1", "x2"], df, sampling_var="D", area_var="group")
intermediate = hbm_flex("y", ["x1", "x2"], df, family="gaussian",
                        sampling_var="D", area_var="group")
model = create_model("y ~ x1 + x2", family="gaussian", data=df,
                     group="group", sampling_var="D")

for m in (beginner, intermediate, model):
    print(m.formula)
```

```{testoutput}
y ~ x1 + x2 + (1|group)
y ~ x1 + x2 + (1|group)
y ~ x1 + x2 + (1|group)
```

The area identifier (`area_var=` or `group=`) adds the area random effect `(1|group)`.

Each family needs its own survey-design columns:

| Family | Beginner interface | Arguments |
| --- | --- | --- |
| Gaussian | `hbm_gaussian` | `sampling_var=`, the sampling variance |
| Beta | `hbm_beta` | `n=` and `deff=`, the sample size and design effect |
| Binomial | `hbm_binomial` | `trials=`, the number of trials |

Every auxiliary variable must be a column name. Compute transformations and interactions
in the DataFrame first:

```{testcode}
import numpy as np

shifted = df.assign(log_x3=np.log(df["x3"] - df["x3"].min() + 1))
print(hbm_gaussian("y", ["x1", "log_x3"], shifted, sampling_var="D",
                   area_var="group").formula)
```

```{testoutput}
y ~ x1 + log_x3 + (1|group)
```

### Set priors

Without `priors=`, Bambi's default priors are used; they are scaled to the data. To set your
own, pass a `Prior` per term:

```{testcode}
from hbsaemp import Prior

informed = hbm_gaussian("y", ["x1", "x2"], df, sampling_var="D", area_var="group",
                        priors={"x1": Prior("Normal", mu=0, sigma=1)})
```

{doc}`03-check-priors` shows whether the priors are reasonable.

### Pin a known parameter

`fixed_params=` fixes a parameter to a known value instead of estimating it: `{"sigma": ...}`
for Gaussian, `{"kappa": ...}` for Beta. The value is a column name or a number, on the
parameter's own scale:

```{testcode}
pinned = create_model("y ~ x1 + x2", family="gaussian", data=df, group="group",
                      fixed_params={"sigma": 2.0})
clean = pinned.check_data()
print([c for c in clean.columns if c not in df.columns])
```

```{testoutput}
['hbsaemp_sigma_fixed']
```

`sampling_var=` already pins `sigma`, so a Gaussian model takes one or the other.

## Check it worked

`summary()` describes the model before fitting:

```{testcode}
print(model.summary())
```

```{testoutput}
GaussianModel [not fitted]
  Formula : y ~ x1 + x2 + (1|group)
  Family  : gaussian
  n       : 30
  Config  : draws=1000, chains=4
```

- `Formula` is the formula that will be fitted.
- `n` is the number of areas left after rows with missing values are dropped.
- `Config` holds the sampler settings.

`check_data()` runs the data pipeline and returns the frame the model will use:

```{testcode}
clean = model.check_data()
print(clean.shape)
print([c for c in clean.columns if c not in df.columns])
```

```{testoutput}
(30, 10)
['log_sqrt_D']
```

`log_sqrt_D` is the Fay-Herriot offset computed from `sampling_var`.

## If it fails

Each of these errors is raised before any sampling.

**The family is unknown.**

```{testcode}
from hbsaemp import ModelRegistryError

try:
    create_model("y ~ x1", family="poisson", data=df)
except ModelRegistryError as err:
    print(err.message)
```

```{testoutput}
Unknown family 'poisson'. Registered families: ['beta', 'binomial', 'gaussian'].
```

**A formula column is not in the data.** `check_data()` and `fit()` raise it, and
`err.context` lists the missing names:

```{testcode}
from hbsaemp import DataValidationError

try:
    create_model("y ~ nope", family="gaussian", data=df).check_data()
except DataValidationError as err:
    print(err.context)
```

```{testoutput}
{'missing_columns': ['nope']}
```

**An argument belongs to another family.**

```{testcode}
try:
    create_model("y ~ x1", family="gaussian", data=df, trials="n")
except ValueError as err:
    print(err)
```

```{testoutput}
Argument(s) not valid for family='gaussian': ['trials']. Valid for 'gaussian': ['link', 'sampling_var'].
```

**A formula term is not a column name.** With `group=`, construction raises it. Without
`group=`, `check_data()` or `fit()` does.

```{testcode}
from hbsaemp import FormulaError

try:
    create_model("y ~ np.log(x1)", family="gaussian", data=df, group="group")
except FormulaError as err:
    print(err.formula)
```

```{testoutput}
y ~ np.log(x1)
```

**A parameter is pinned twice.**

```{testcode}
try:
    create_model("y ~ x1", family="gaussian", data=df, group="group",
                 sampling_var="D", fixed_params={"sigma": 2.0})
except ValueError as err:
    print(err)
```

```{testoutput}
'sigma' is pinned twice: once through ['sampling_var'] and once through fixed_params['sigma']. Pass one or the other.
```

## Related

- Why there are three interfaces: {doc}`../explanation/three-tier-api`.
- Which columns each family accepts: {doc}`../explanation/family-registry`.
