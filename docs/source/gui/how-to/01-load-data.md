# 1 · Load data

Get your data into the app, from either your own file or a built-in example, and
confirm it landed correctly before moving on.

## Before you start

You need the app running — see {doc}`../index` if you haven't started it yet. This
guide picks up on the **Data** tab, which is where the app opens by default.

You need one of two things:

- A **CSV file** of your own, structured with one row per small area:
  - the first row is a header with column names;
  - one column holds the response you want to estimate (the direct/survey estimate);
  - one or more columns hold auxiliary variables (predictors from outside the survey);
  - optionally, a column identifying each area.
- Or nothing at all — the app ships with four built-in example datasets you can load
  instead, useful for trying the workflow before bringing your own data.

```{note}
CSV files must use a comma (`,`) as the column separator and a period (`.`) as the
decimal mark. A file exported with a different locale (comma as decimal, semicolon
as separator) will either fail to load or load with the wrong columns.
```

```{note}
There is an upload size limit — 50 MB by default, though whoever started the app may
have configured a different value. A file over the limit is rejected before it is
read, so it never partially loads.
```

Loading data does **not** remove incomplete rows. Rows with missing values in the
columns you end up using are only dropped later, at model-fitting time — so the
missing-value count you see here is informational, not a filter.

## Do it

::::{tab-set}

:::{tab-item} Upload your own file
1. On the **Data** tab, find the **Upload File / Load Built-in Dataset** card.
2. Click **Choose File** and choose your file.
3. The app reads it immediately — no separate "load" click is needed for an upload.
:::

:::{tab-item} Use a built-in dataset
1. On the **Data** tab, find the **Upload File / Load Built-in Dataset** card.
2. Open the **Or load a built-in dataset** dropdown and pick one:

   | Dataset | Fits with family |
   | --- | --- |
   | `data_fhnorm` | Gaussian (Fay-Herriot) |
   | `data_betalogitnorm` | Beta |
   | `data_binlogitnorm` | Binomial |
   | `data_lnln` | reserved for a future model type — not yet fittable in this app |

3. Click **Load Dataset**.
:::

::::

```{figure} images/01-upload-or-load-dataset.png
:alt: Upload File / Load Built-in Dataset card on the Data tab
:width: 600px

The **Upload File / Load Built-in Dataset** card on the Data tab.
```

Either way, a green banner confirms success and tells you the next stop is the
**Data Exploration** tab.

## Check it worked

Three cards below the upload card update automatically once data is loaded:

- **Summary Dataset** — total rows, total columns, and total missing values as one
  line. Use this as your first sanity check: if the row or column count is not what
  you expected, you likely uploaded the wrong file.
- **Variable List** — every column name, as a row of badges. Confirm your response
  column and auxiliary columns are all present and spelled the way you expect —
  you'll pick them by name in the Model tab later.
- **Data Preview** — the actual table, paginated 10 rows at a time. Spot-check a few
  rows for obviously wrong values (shifted columns, text in a numeric field).

```{figure} images/01-summary-variablelist-preview.png
:alt: Summary Dataset, Variable List, and Data Preview cards after data loads successfully
:width: 600px

**Summary Dataset**, **Variable List**, and **Data Preview** after data has
loaded successfully.
```

If all three look right, you're ready to move to exploration or straight to modeling.

## If it fails

**"File too large."**
Your file exceeds the upload limit. Split it, remove unneeded columns, or ask
whoever runs the app to raise `max_upload_mb`.

**"Could not read CSV file."**
The file isn't valid CSV, or uses a separator/decimal mark the app doesn't expect.
Open it in a spreadsheet program and re-save as standard comma-separated CSV.

**"Please select a built-in dataset first."**
You clicked **Load Dataset** without picking one from the dropdown. Pick a dataset,
then click the button.

**"Could not load dataset."**
This points to a problem in the app's own bundled data rather than anything you did
— worth reporting rather than retrying with a different file.

## Related

For why the app defers dropping incomplete rows to fitting time, instead of cleaning
them here, see {doc}`../explanation/index`.