# 2 · Data Understanding

## Get the data

This case study uses a cleaned extract of the 2024 SUSENAS CHE indicator,
joined with PODES 2024 village-census variables — 514 districts
(*kabupaten/kota*) across Indonesia, 38 columns.

{download}`Download data_case_study_che.csv <data/data_case_study_che.csv>`

Download the CSV above and keep it somewhere you can find it — the next
stage ({doc}`03-data-preparation`) uploads this exact file to the app's
**Data** tab.

## What's in it

Two identifiers, four columns from the direct survey estimate (SUSENAS),
and 32 auxiliary predictors from the village census (PODES), grouped by
what they measure.

{download}`Download variable_dictionary.csv <data/variable_dictionary.csv>`
— every one of the 38 columns, with its source and what it means. The
table below covers only the columns this case study actually uses; open
the dictionary for the rest.

`````{list-table}
:header-rows: 1
:widths: 15 15 70

* - Column
  - Source
  - What it means
* - `IDKABKOT`
  - SUSENAS
  - BPS district code — a unique identifier for each *kabupaten/kota*.
* - `NAMA_KAB`
  - SUSENAS
  - District name.
* - `y`
  - SUSENAS
  - Number of surveyed households classified as CHE (health spending over
    10% of total household spending) in that district.
* - `n`
  - SUSENAS
  - Number of surveyed households in that district — the direct sample
    size behind `y`.
* - `est_prop`
  - SUSENAS
  - Direct estimate of the CHE proportion for the district (survey-weighted
    `y` / `n`). This is the response variable — what the whole case study
    is trying to estimate more reliably.
* - `rse_prop`
  - SUSENAS
  - Relative standard error of `est_prop`. Missing (`NA`) wherever
    `est_prop` is 0 — 24 districts in this dataset.
* - `proporsi_R704IK2`
  - PODES
  - Share of villages in the district with a practicing midwife
    (*tempat praktik bidan*) — a primary-care access proxy. Used as
    auxiliary variable **X1** in the model.
* - `proporsi_R704JK2`
  - PODES
  - Share of villages in the district with a village health post
    (*Pos Kesehatan Desa* / poskesdes) — another primary-care access proxy.
    Used as auxiliary variable **X2** in the model.
* - `proporsi_R805I`
  - PODES
  - Share of villages in the district with a resident who has multiple
    disabilities (*penyandang cacat ganda*) — a proxy for household
    vulnerability and likely healthcare cost burden. Used as auxiliary
    variable **X3** in the model.
`````

```{note}
The full file carries 32 PODES columns in total — health-facility access
(`proporsi_R704*`), Posyandu/Posbindu activity (`proporsi_R705*`), resident
health workers (`proporsi_R706*`), and residents' disability types
(`proporsi_R805*`). Only `proporsi_R704IK2`, `proporsi_R704JK2`, and
`proporsi_R805I` are used as auxiliary variables here — the three found to
relate most closely to `est_prop` — which keeps this case study's model
simple enough to walk through end to end. The rest stay in the file so you
can explore them yourself on the **Data Exploration** tab; see the
dictionary CSV above for what each one means.
```

## A first look

Before opening the app, it's worth knowing what to expect once the data is
loaded:

- `est_prop` ranges from 0 up to about 0.14 (14%) — no district in this
  extract is anywhere near the national 2.7% figure being typical; the
  spread itself is part of the story.
- `rse_prop` has 24 missing values, corresponding exactly to the 24
  districts where `est_prop` is 0 — a data quality quirk worth remembering
  once diagnostics are in view later, not a loading error.
- `proporsi_R704IK2`, `proporsi_R704JK2`, and `proporsi_R805I` are all
  proportions between 0 and 1, already on a comparable scale to each other
  — no rescaling needed before modeling.

## Load it into the app

With `data_case_study_che.csv` downloaded, open the app and load it in.

1. Open the **Data** tab.
2. Click **Upload CSV File** and select `data_case_study_che.csv`.

   ```{figure} images/02-upload-che-data.png
   :alt: Uploading data_case_study_che.csv on the Data tab
   :width: 600px

   Uploading `data_case_study_che.csv` on the Data tab.
   ```

3. Check the **Summary Dataset** card: it should read **514** total rows,
   **38** total columns, and **24** missing values — exactly the 24 `NA`
   values in `rse_prop` noted above, and nowhere else in the file. If any
   of these three numbers is different, the upload picked up the wrong
   file or an edited copy — re-download and try again rather than
   continuing.

   ```{figure} images/02-summary-dataset-check.png
   :alt: Summary Dataset card showing 514 rows, 38 columns, 24 missing values
   :width: 600px

   **Summary Dataset** reading 514 rows, 38 columns, 24 missing values.
   ```

4. Check the **Variable List** badges against the dictionary above — you
   should see `IDKABKOT`, `NAMA_KAB`, `y`, `n`, `est_prop`, `rse_prop`, and
   32 `proporsi_*` columns, 38 badges in total.

   ```{figure} images/02-variable-list-check.png
   :alt: Variable List badges matching the 38 expected columns
   :width: 600px

   **Variable List** showing all 38 column names as badges.
   ```

5. Check the **Data Preview** table: the first rows should show district
   names under `NAMA_KAB` (e.g. `SIMEULUE`, `ACEH SINGKIL`), whole numbers
   under `y` and `n`, and small decimals under `est_prop` and the
   `proporsi_*` columns.

   ```{figure} images/02-data-preview-check.png
   :alt: Data Preview table showing the first rows of the CHE dataset
   :width: 600px

   **Data Preview** showing the first rows.
   ```

Once all three match, move on to the **Data Exploration** tab below before
moving to {doc}`03-data-preparation`.

## Explore before modeling

With the data loaded, take a look at it on the **Data Exploration** tab
before touching the Modeling tab — the shapes below are worth remembering
once the model's estimates are in view later.

6. Check **Summary Statistics**.

   ```{figure} images/02-summary-statistics.png
   :alt: Summary Statistics table for all 38 columns
   :width: 700px

   The **Summary Statistics** table, one row per numeric column.
   ```

   Check the **Min** and **Max** columns for the `proporsi_*` predictors
   (including `proporsi_R704IK2`, `proporsi_R704JK2`, and `proporsi_R805I`):
   every one of them should read **Min 0.0** and **Max** no higher than
   **1.0** — they're proportions of villages, so anything outside 0–1 would
   mean a data problem, not a modeling one. If any predictor's Max reads
   above 1.0, stop and re-check the upload before continuing.

7. Check **Histogram and Boxplot**.

   Six variables matter most for this case study: the response `y` and its
   sample size `n`, the direct estimate `est_prop`, and the three predictors
   used later as X1, X2, and X3.

   **`est_prop`** — the response. Strongly right-skewed: most districts sit
   below 0.02, with a long tail out to about 0.14 and several boxplot
   outliers above the upper whisker. This is exactly the shape that makes
   direct estimation unreliable at the low end — many districts near zero,
   a handful pulling the average up.

   ```{figure} images/02-hist-box-est_prop.png
   :alt: Histogram and boxplot of est_prop
   :width: 600px

   Histogram and boxplot of `est_prop`.
   ```

   **`n`** — the survey sample size. Roughly bell-shaped and much less skewed
   than `est_prop`, centered around 2,200–2,400 households, with a handful of
   districts sampled above 3,500 or below 1,500. This is closer to what a
   designed survey sample is supposed to look like — the skew problem lives
   in the response, not the sample sizes.

   ```{figure} images/02-hist-box-n.png
   :alt: Histogram and boxplot of n
   :width: 600px

   Histogram and boxplot of `n`.
   ```

   **`y`** — CHE household counts. The same right-skewed shape as `est_prop`,
   which makes sense: `y` is the raw count behind that proportion, so
   whatever pulls `est_prop` up in a handful of districts pulls `y` up too.

   ```{figure} images/02-hist-box-y.png
   :alt: Histogram and boxplot of y
   :width: 600px

   Histogram and boxplot of `y`.
   ```

   **`proporsi_R704IK2` (X1 — midwife practice coverage)** — not skewed the
   way `est_prop` is; the histogram is closer to U-shaped, with a spike of
   districts near 0 (little to no midwife-practice coverage), a dip in the
   middle, and another rise toward 1.0. The boxplot has no outliers and
   whiskers spanning the full 0–1 range — this predictor genuinely varies
   across its whole range rather than clustering with a few extreme
   districts.

   ```{figure} images/02-hist-box-x1.png
   :alt: Histogram and boxplot of proporsi_R704IK2 (X1)
   :width: 600px

   Histogram and boxplot of `proporsi_R704IK2` (X1).
   ```

   **`proporsi_R704JK2` (X2 — poskesdes coverage)** — a similar shape to X1:
   a spike near 0, then a fairly flat spread from about 0.1 up to 1.0. Also
   no boxplot outliers, also spanning the full range.

   ```{figure} images/02-hist-box-x2.png
   :alt: Histogram and boxplot of proporsi_R704JK2 (X2)
   :width: 600px

   Histogram and boxplot of `proporsi_R704JK2` (X2).
   ```

   **`proporsi_R805I` (X3 — multiple-disability prevalence)** — back to a
   strongly right-skewed shape, closer to `est_prop` and `y` than to X1/X2:
   most districts sit below 0.15, with a long tail out to about 0.9 and many
   boxplot outliers above the upper whisker.

   ```{figure} images/02-hist-box-x3.png
   :alt: Histogram and boxplot of proporsi_R805I (X3)
   :width: 600px

   Histogram and boxplot of `proporsi_R805I` (X3).
   ```

   ```{note}
   X1 and X2 spread fairly evenly across 0–1; X3 is concentrated near 0 with
   a long tail, much like the response itself. Keep that contrast in mind —
   it resurfaces in {doc}`03-data-preparation` when deciding how each
   predictor enters the model.
   ```

8. Check **Scatter & Correlation**.

   The Scatter & Correlation sub-tab reports correlation against whichever
   Y-axis transform you pick — for this case study, pick **Logit** for the
   Y-axis Variable (`est_prop`), not **None**.

   ```{important}
   Why the logit, not raw `est_prop`? The model this case study builds later
   is a **logit-normal** HB model: it puts a linear predictor on the
   *logit* of the proportion, not on the proportion itself — that's what
   "logit-normal" means. Pearson's r specifically measures *linear*
   relationship. A predictor can look weak against raw `est_prop` (which is
   squeezed into [0, 1] and flattens out near the boundaries) yet look
   substantially stronger against `est_prop_logit`, because the logit
   transform is exactly what un-squeezes that boundary compression. Checking
   correlation on the scale the model actually assumes linearity on is the
   check that's relevant to whether a predictor will help the regression —
   checking it on the raw scale can understate a predictor the model would
   otherwise use well, or overstate one it wouldn't.
   ```

   Set the **X-axis Variable** to each predictor in turn and **Y-axis
   Variable** to `est_prop` with transform **Logit**. For X1, X2, and X3:

   ```{note}
   **How to read the four metrics.** Each measures a different kind of
   relationship, which is why they don't always agree:
   - **Pearson's r** (−1 to 1) — straight-line relationship only. Two
     variables with a strong curved relationship can still score low here.
   - **Spearman's rho** (−1 to 1) — any consistently increasing or
     decreasing relationship ("monotonic"), straight or curved. When it's
     well above Pearson's r, the relationship is real but not a straight
     line.
   - **Chatterjee's Xi** (0 to 1) — any relationship where X predicts Y as a
     function, even a non-monotonic one (rising then falling, for example).
     It's asymmetric: correlating X with Y can give a different Xi than Y
     with X.
   - **Distance Correlation** (0 to 1, no sign) — any dependence at all
     between the two variables, linear or not, monotonic or not. The most
     general of the four, at the cost of not saying which direction the
     relationship runs.

   None of the four is simply "better" — reading them together tells you not
   just *whether* two variables relate, but *what shape* that relationship
   takes.
   ```

   ```{figure} images/02-scatter-x1.png
   :alt: Correlation and scatter plot of proporsi_R704IK2 (X1) against est_prop_logit
   :width: 600px

   `proporsi_R704IK2` (X1) vs `est_prop_logit`.
   ```

   ```{figure} images/02-scatter-x2.png
   :alt: Correlation and scatter plot of proporsi_R704JK2 (X2) against est_prop_logit
   :width: 600px

   `proporsi_R704JK2` (X2) vs `est_prop_logit`.
   ```

   ```{figure} images/02-scatter-x3.png
   :alt: Correlation and scatter plot of proporsi_R805I (X3) against est_prop_logit
   :width: 600px

   `proporsi_R805I` (X3) vs `est_prop_logit`.
   ```

   `````{list-table}
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Predictor
     - Pearson's r
     - Spearman's rho
     - Chatterjee's Xi
     - Distance Correlation
   * - X1 — `proporsi_R704IK2`
     - 0.496
     - 0.680
     - 0.333
     - 0.579
   * - X2 — `proporsi_R704JK2`
     - 0.294
     - 0.239
     - 0.134
     - 0.291
   * - X3 — `proporsi_R805I`
     - 0.280
     - 0.505
     - 0.179
     - 0.375
   `````

   All three p-values round to 0 — with 514 districts, even a modest
   correlation is unlikely to be chance. What differs is the *size* of the
   relationship, not whether one exists.

   **X1** is clearly the strongest of the three, on every metric — its
   scatter plot shows the tightest, most visibly upward-sloping cloud of the
   three, and its Spearman's rho (0.680) well above its Pearson's r (0.496)
   says the relationship is monotonic but somewhat curved rather than a
   straight line.

   **X2** is the weakest here, and unusually its Spearman's rho (0.239) is
   *lower* than its Pearson's r (0.294) — the opposite pattern from X1 and
   X3. Its scatter plot is the noisiest of the three: points are spread
   widely at every X value, with only a shallow overall trend — the logit
   transform doesn't rescue a relationship that's genuinely diffuse to begin
   with.

   **X3** sits in between: a modest Pearson's r (0.280) but a substantially
   higher Spearman's rho (0.505), meaning the relationship is real and
   monotonic but non-linear — visible in the scatter as a cluster of
   low-`proporsi_R805I` districts with widely scattered logit values, and a
   smaller, more consistently high-logit group at the upper end of
   `proporsi_R805I`.

   **So — are X1, X2, and X3 fit to be predictors?** All three are
   statistically real (p ≈ 0) and point the expected direction — more health
   facility access or a higher share of vulnerable residents both associate
   with a higher CHE burden, which matches the Business Understanding
   framing. But "statistically real" is not the same as "individually
   strong": only X1 would stand on its own as a solid predictor by
   conventional thresholds; X2 and X3 are worth keeping as *additional*
   signal alongside X1 and the model's district-level random effect, not as
   predictors expected to carry the model by themselves. That's a realistic
   outcome for auxiliary survey data, not a flaw in the case study — it's
   exactly the situation small-area estimation is built for, borrowing a
   little strength from several imperfect sources rather than needing one
   perfect one.

   ````{note}
   **A candidate worth remembering for later.** `proporsi_R805G` — the share
   of villages with a resident who has recovered from mental illness
   (*eks sakit jiwa*) — correlates with `est_prop_logit` at **Pearson's r =
   0.511, Spearman's rho = 0.504**, putting it on par with X1 and ahead of
   both X2 and X3.

   ```{figure} images/02-scatter-r805g.png
   :alt: Correlation and scatter plot of proporsi_R805G against est_prop_logit
   :width: 600px

   `proporsi_R805G` vs `est_prop_logit`.
   ```

   It isn't used as an auxiliary variable in this case study, to keep the
   model to three predictors — but if a future revision needs a fourth
   predictor, or a replacement for X2 or X3, `proporsi_R805G` is the first
   place to look, ahead of anything else in the dictionary that wasn't
   already tried.
   ````