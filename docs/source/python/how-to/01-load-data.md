# 1 · Load the data

Get a table with one row per small area into a `pandas.DataFrame`.

## Before you start

A model reads four kinds of column:

| Column | In `data_fhnorm` |
| --- | --- |
| Response: the direct estimate per area | `y` |
| Auxiliary variables | `x1`, `x2`, `x3` |
| Area identifier | `group` |
| Survey-design columns required by the family | `D`, the sampling variance |

The survey-design columns depend on the family. Gaussian needs the sampling variance,
Beta needs the sample size and the design effect, and Binomial needs the number of trials.

## Do it

Load a bundled dataset by name:

```{testcode}
from hbsaemp import load_dataset

df = load_dataset("data_fhnorm")
print(df.shape)
print(list(df.columns))
```

```{testoutput}
(30, 9)
['y', 'D', 'x1', 'x2', 'x3', 'theta_true', 'u', 'group', 'sre']
```

Two lists name the bundled datasets:

```{testcode}
from hbsaemp import AVAILABLE_DATASETS, REAL_DATASETS

print(AVAILABLE_DATASETS)
print(REAL_DATASETS)
```

```{testoutput}
['data_fhnorm', 'data_betalogitnorm', 'data_binlogitnorm', 'data_lnln']
['susenas2023_papua', 'podes2021_papua']
```

The synthetic datasets in `AVAILABLE_DATASETS` are ready to model. The real Papua datasets
in `REAL_DATASETS` load as published and need preparing first, as shown in
{doc}`../tutorials/case-study-papua`.

For your own data, read the file with pandas:

```python
import pandas as pd

df = pd.read_csv("areas.csv")
```

## Check it worked

Count the missing values:

```{testcode}
print(df.isna().sum().sum())
```

```{testoutput}
0
```

A row with a missing value in any column the model uses is dropped when the model is built.
Real data often has some:

```{testcode}
papua = load_dataset("susenas2023_papua")
print(papua[["morbidity_rate", "se"]].isna().sum().to_dict())
```

```{testoutput}
{'morbidity_rate': 1, 'se': 1}
```

The synthetic datasets also contain the values used to generate them, such as `theta_true`
and `u`. Use them only to check estimates against the truth, never as auxiliary variables.

## If it fails

**The dataset name is unknown.** The message lists every valid name:

```{testcode}
try:
    load_dataset("data_fh")
except ValueError as err:
    print(err)
```

```{testoutput}
Unknown dataset 'data_fh'. Available: ['data_fhnorm', 'data_betalogitnorm', 'data_binlogitnorm', 'data_lnln', 'susenas2023_papua', 'podes2021_papua']
```

**An area spans several rows.** hbsaemp models area-level data. Aggregate unit-level
records to one row per area first, for example with `df.groupby(...)`. Otherwise
{doc}`09-estimate-areas` returns one row per observation and logs a warning.

## Related

Every column of every bundled dataset is described in {doc}`../reference/datasets`.
