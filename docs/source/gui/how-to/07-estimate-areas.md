# 7 · Estimate the areas

Turn the fitted model into a table of per-area estimates, optionally including
areas that weren't sampled, and download the result.

## Before you start

You need a fitted model — see {doc}`04-fit-model`. This page covers the
**Results** tab's **SAE Estimation** sub-tab. This is the step whose output
you'll likely publish, so check {doc}`05-check-convergence` first.

## Do it

1. Open **Results → SAE Estimation**.

   ```{figure} images/07-step1-sae-estimation-subtab.png
   :alt: SAE Estimation sub-tab
   :width: 600px

   The **SAE Estimation** sub-tab.
   ```

2. Adjust the **Credible interval (ci_prob)** if you don't want the default
   0.95.

   ```{figure} images/07-step2-ci-prob-field.png
   :alt: Credible interval (ci_prob) field
   :width: 600px

   The **Credible interval (ci_prob)** field.
   ```

3. **Optional — estimate for new/unsampled areas.** Check
   "Estimate for new/unsampled areas" and upload a CSV of predictors (plus an
   area column) for areas with no survey data — no response column needed.
   This replaces the training data for this run only; it does not change
   what's loaded on the Data tab.

   ```{figure} images/07-step3-new-areas-upload.png
   :alt: Estimate for new/unsampled areas checkbox and CSV upload
   :width: 600px

   The **"Estimate for new/unsampled areas"** checkbox and its CSV upload.
   ```

4. Click **Run SAE Estimation**.

   ```{figure} images/07-step4-run-sae-button.png
   :alt: Run SAE Estimation button
   :width: 600px

   The **Run SAE Estimation** button.
   ```

5. Review the results table and any notes below the status message.

   ```{figure} images/07-step5-results-table-and-notes.png
   :alt: SAE results table and notes below the status message
   :width: 600px

   The results table and the notes below the status message.
   ```

6. Click **Download CSV of SAE Results** to save the table.

   ```{figure} images/07-step6-download-csv-button.png
   :alt: Download CSV of SAE Results button
   :width: 600px

   The **Download CSV of SAE Results** button.
   ```

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