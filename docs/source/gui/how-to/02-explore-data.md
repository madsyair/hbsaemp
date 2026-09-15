# 2 · Explore data

Look at your variables before you commit to a model — their distributions, and how
strongly pairs of them relate to each other.

## Before you start

You need data already loaded — see {doc}`01-load-data`. This tab only ever shows
**numeric** columns: any text or category column you loaded (an area-name identifier,
for example) will not appear in any dropdown here, since none of the summaries or
plots on this tab apply to it.

If you skip this tab, nothing is lost — the Model tab works whether or not you've
visited Data Exploration. Use it when you want a look at the data before deciding
which auxiliary variables and transformations make sense.

## Do it

The tab has three sub-tabs.

::::{tab-set}

:::{tab-item} Summary Statistics
Open on landing. One row per numeric variable: Min, 1st Qu., Median, Mean, 3rd Qu.,
Max. No configuration needed — it covers every numeric column at once.

```{figure} images/02-summary-statistics.png
:alt: Summary Statistics table
:width: 600px

The **Summary Statistics** table.
```
:::

:::{tab-item} Visualize Distribution
Two independent cards, side by side:

1. **Histogram** — pick a **Variable**, then adjust **Number of bins** (1–100) with
   the slider. A kernel density curve is overlaid automatically.

   ```{figure} images/02-histogram-controls.png
   :alt: Variable selector and Number of bins slider for the Histogram card
   :width: 600px

   The **Variable** selector and **Number of bins** slider for the Histogram card.
   ```

   ```{figure} images/02-histogram-plot.png
   :alt: Histogram with kernel density curve overlaid
   :width: 600px

   The resulting histogram, with a kernel density curve overlaid.
   ```

2. **Boxplot** — pick a **Variable**. Independent from the histogram's variable
   choice, so you can compare two different columns at a glance.

   ```{figure} images/02-boxplot-controls.png
   :alt: Variable selector for the Boxplot card
   :width: 600px

   The **Variable** selector for the Boxplot card.
   ```

   ```{figure} images/02-boxplot-plot.png
   :alt: Boxplot for the selected variable
   :width: 600px

   The resulting boxplot.
   ```
:::

:::{tab-item} Scatter & Correlation
1. Pick an **X-axis Variable** and a **Y-axis Variable**.
2. Optionally apply a transformation to either axis:
   - X: None, Log, Z-score.
   - Y: None, Log, Logit, Z-score.

   ```{figure} images/02-scatter-controls.png
   :alt: X-axis/Y-axis variable selectors and transformation dropdowns
   :width: 600px

   The **X-axis**/**Y-axis Variable** selectors and their transformation dropdowns.
   ```

3. Read the correlation table and scatter plot, both of which update immediately.

   ```{figure} images/02-correlation-table.png
   :alt: Correlation table with Pearson, Spearman, Chatterjee's Xi, and Distance Correlation
   :width: 600px

   The correlation table — Pearson's r, Spearman's rho, Chatterjee's Xi, and
   Distance Correlation side by side.
   ```

   ```{figure} images/02-scatter-plot.png
   :alt: Scatter plot with trendline for the selected X and Y variables
   :width: 600px

   The scatter plot, with a trendline drawn for the selected X and Y variables.
   ```
:::

::::

Every plot and table on this tab reacts live to your selections — there's no
"apply" button to click.

## Check it worked

- **Summary Statistics**: check that the Min/Max for each variable are within the
  range you expect for that measurement. A Max far larger than the rest of the
  column usually means an outlier or a units mismatch worth investigating before
  modeling.
- **Histogram/Boxplot**: look for skew or outliers now, since some families
  (Beta, Binomial) assume a bounded response and will reject values outside their
  valid range at model-fitting time.
- **Scatter & Correlation**: the table reports four measures side by side —
  Pearson's r and Spearman's rho for the usual linear/monotonic relationships,
  plus Chatterjee's Xi and Distance Correlation, which can catch a dependency
  between two variables even when it isn't linear or monotonic. A trendline is
  drawn on the scatter plot to help you judge the fit visually.

## If it fails

**A variable I expect to see isn't in any dropdown.**
Only numeric columns are shown. If a column you need was loaded as text — a
numeric-looking ID column with leading zeros, for instance — it won't appear here,
though it can still be used as a `group` (area identifier) in the Model tab.

**The tab is empty, or dropdowns have no options.**
No data is loaded yet. Go back to {doc}`01-load-data`.

**The correlation table or scatter plot doesn't update after picking a variable.**
This usually means the chosen column has too few non-missing values to compute a
correlation. Check its missing-value count first — the Summary Statistics table
counts only present values, so a variable that's mostly `NaN` will still show
there, just with a smaller effective sample.

## Related

For why the app reports Chatterjee's Xi and Distance Correlation alongside the more
familiar Pearson and Spearman, see {doc}`../explanation/index`.