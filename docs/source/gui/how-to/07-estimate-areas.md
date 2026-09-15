# 7 · Estimate the areas

Turn the fitted model into a table of per-area estimates, optionally including
areas that weren't sampled, and download the result.

## Before you start

You need a fitted model — see {doc}`04-fit-model`. This page covers the
**Results** tab's **SAE Estimation** sub-tab. This is the step whose output
you'll likely publish, so check {doc}`05-check-convergence` first.

## Do it

1. Open **Results → SAE Estimation**.
2. Adjust the **Credible interval (ci_prob)** if you don't want the default
   0.95.
3. **Optional — estimate for new/unsampled areas.** Check
   "Estimate for new/unsampled areas" and upload a CSV of predictors (plus an
   area column) for areas with no survey data — no response column needed.
   This replaces the training data for this run only; it does not change
   what's loaded on the Data tab.
4. Click **Run SAE Estimation**.
5. Review the results table and any notes below the status message.
6. Click **Download CSV of SAE Results** to save the table.

## Check it worked

- A green message reports the run's **Mean RSE** and **Mean MSE** across
  areas.
- If you supplied new-area data, an `area_type` column marks each row
  `sampled` or `non-sampled`.
- Notes appear automatically for things worth knowing about the specific
  run: repeated area labels (rows are per observation, not aggregated), areas
  with undefined RSE (mean of 0), or wider intervals for non-sampled areas
  (their area effect comes from the distribution of other areas, not direct
  data).

## If it fails

**"Model has not been fitted."**
Fit a model first — see {doc}`04-fit-model`.

**"No file uploaded" (with "estimate for new/unsampled areas" checked).**
Upload a CSV, or uncheck the box to estimate on the original data instead.

**"Could not read CSV file."**
The uploaded file isn't valid CSV — check its separator and encoding.

**"Estimation failed."**
The message names what the backend rejected — often a missing predictor
column in the new-area data. Fix the CSV and rerun.

## Related

For what the credible-interval width means for non-sampled areas specifically,
see {doc}`../explanation/index` (a dedicated page lands with the remaining
explanation pages).