# 1 · Specify a model

State which column is the response, which carry auxiliary information, and how areas
are identified.

## Before you start

You need a `pandas.DataFrame` with **one row per small area**. Three kinds of column
matter: the response you want to estimate, the auxiliary variables that carry
information from outside the survey, and the column identifying the area. Which extra
columns a family requires depends on the family — the Gaussian Fay-Herriot model needs
a known sampling variance, Beta needs sample size and design effect, Binomial needs the
number of trials.

Specifying a model does **not** sample. Nothing here is expensive, and nothing here
imports the MCMC backend, so mistakes surface in milliseconds rather than after a long
wait.

## Do it

```{testcode}
from hbsaemp import create_model, load_dataset

df = load_dataset("data_fhnorm")

model = create_model(
    "y ~ x1 + x2",
    family="gaussian",
    data=df,
    group="group",
    sampling_var="D",
)
print(model.formula)
```

```{testoutput}
y ~ x1 + x2 + (1|group)
```

Note what the formula became. You wrote `y ~ x1 + x2`, and `group="group"` added the
area random effect `(1|group)` for you. Writing that term yourself is equally valid and
is respected rather than duplicated.

Every term on the right of `~` must be a **bare column name**. Transformations belong in
the DataFrame, computed before the model sees it, and then referred to by their new
column name:

```{testcode}
import numpy as np

shifted = df.assign(log_x3=np.log(df["x3"] - df["x3"].min() + 1))
transformed = create_model(
    "y ~ x1 + log_x3",
    family="gaussian",
    data=shifted,
    group="group",
    sampling_var="D",
)
print(transformed.formula)
```

```{testoutput}
y ~ x1 + log_x3 + (1|group)
```

## Check it worked

Three properties answer "did it understand me?", and none of them sample:

```{testcode}
print(model.response_name)
print(model.family)
print(model.is_fitted)
```

```{testoutput}
y
gaussian
False
```

Then run the data pipeline on its own. `check_data()` validates the frame and applies
the family's transformations, returning the frame the model will actually use — so you
find out about a bad column now rather than after sampling starts:

```{testcode}
clean = model.check_data()
print(clean.shape)
print([c for c in clean.columns if c not in df.columns])
```

```{testoutput}
(30, 10)
['log_sqrt_D']
```

The extra column is the Fay-Herriot offset the Gaussian family derives from
`sampling_var`. You never add it yourself.

## If it fails

**The family name is not recognised.**

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

**A column in the formula is not in the data.** This one surfaces from the pipeline, not
from construction:

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

The message itself also lists every column that *is* available, which is usually enough
to spot a typo immediately. Read `.context` rather than `.column` here: a single
`.column` cannot hold several missing names at once, so it stays `None` and the list
goes into the context dict.

**An argument belongs to a different family.** Each family accepts only its own extras,
so a misplaced one is refused rather than quietly ignored:

```{testcode}
try:
    create_model("y ~ x1", family="gaussian", data=df, trials="n")
except ValueError as err:
    print(err)
```

```{testoutput}
Argument(s) not valid for family='gaussian': ['trials']. Valid for 'gaussian': ['link', 'sampling_var'].
```

**A formula term is not a bare column name.** Where this is reported depends on one
detail worth knowing:

| How you call it | When you find out |
| --- | --- |
| `create_model("y ~ np.log(x1)", ...)` without `group=` | construction succeeds; the error arrives at `check_data()` or `fit()` |
| the same call **with** `group=` | raised immediately, during construction |

The difference is that resolving `group=` has to parse the formula, so passing it makes
the check happen sooner. Either way the fix is the same: compute the transformed column
in the DataFrame first, as shown above.

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

Rejecting the term is deliberate. Silently dropping it would hide a predictor from
validation and resurface much later as an obscure error from the modelling backend.

## Related

Which extras a family accepts, and why that list lives in one place rather than being
repeated per model, is explained in {doc}`../explanation/family-registry`.
