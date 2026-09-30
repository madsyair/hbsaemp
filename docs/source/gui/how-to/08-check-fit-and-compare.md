# 8 · Check the fit and compare models

Check that the fitted model reproduces the observed data, then rank the models you saved and optionally switch the active model to one of them.
The posterior predictive check is on the **Modeling** tab. The comparison is on **Results → Model Comparison**.

## Before you start

You need a fitted model whose convergence you have checked. See {doc}`06-check-convergence`. For the comparison you also need at least one model saved. See the **Save Model** step in {doc}`05-fit-model`.
Saved models don't need to have converged. See {doc}`../explanation/convergence` for why the app doesn't check.

## Do it

::::{tab-set}

:::{tab-item} 1. Posterior Predictive Check
On the **Posterior Predictive Check** sub-tab, available once a model is
fitted:

1. Click **Run Posterior Predictive Check**.

   ```{figure} images/04-posterior-run-button.png
   :alt: Run Posterior Predictive Check button
   :width: 600px

   The **Run Posterior Predictive Check** button.
   ```

2. Read the **Posterior Predictive Plot** (does simulated data resemble the
   observed data?) and the **Posterior Predictive Credible Intervals by
   Observation** plot (rootgram).

   ```{figure} images/04-posterior-plots.png
   :alt: Posterior Predictive Plot and Posterior Predictive Credible Intervals by Observation (rootgram) plot
   :width: 600px

   The **Posterior Predictive Plot** and the **Posterior Predictive Credible
   Intervals by Observation** (rootgram) plot.
   ```
:::

:::{tab-item} 2. Compare models
1. Open **Results → Model Comparison**. The table lists every model you've
   saved, by name.

   ```{figure} images/06-step1-model-comparison-table.png
   :alt: Model Comparison sub-tab with the table of saved models
   :width: 600px

   The **Model Comparison** sub-tab, with the table of saved models.
   ```

2. Check the box next to each model you want to compare. A single checked
   model shows its own LOO/ELPD and (if enabled) Bayes Factor; two or more
   also produce a side-by-side ranking.

   ```{figure} images/06-step2-model-checkboxes.png
   :alt: Checkboxes for selecting which saved models to compare
   :width: 600px

   The checkboxes for selecting which saved models to compare.
   ```

3. Optionally check **Include Bayes Factor (Savage-Dickey, per coefficient)**. It costs an extra sampling
   pass per model on top of plain LOO, so it is off by default.

   ```{figure} images/06-step3-bayes-factor-checkbox.png
   :alt: Include Bayes Factor (Savage-Dickey, per coefficient) checkbox
   :width: 600px

   The **Include Bayes Factor (Savage-Dickey, per coefficient)** checkbox.
   ```

4. Click **Compare Selected**.

   ```{figure} images/06-step4-compare-selected-button.png
   :alt: Compare Selected button
   :width: 600px

   The **Compare Selected** button.
   ```

5. Read the **Ranking (LOO / ELPD)** table, the ranking/posterior predictive/
   parameter plots, and the Bayes Factor table if enabled.

   ```{figure} images/06-step5-ranking-table-and-plots.png
   :alt: Ranking (LOO / ELPD) table and comparison plots
   :width: 600px

   The **Ranking (LOO / ELPD)** table and the comparison plots.
   ```

6. To make one of the compared models the active one, pick it from the
   **Select a model to use** dropdown and click **Use Selected Model**.

   ```{figure} images/06-step6-use-selected-model.png
   :alt: Select a model to use dropdown and Use Selected Model button
   :width: 600px

   The **Select a model to use** dropdown and the **Use Selected Model** button.
   ```
:::

::::

## Check it worked

- Posterior check: both plots render. If only one does, the other shows its
  own error box rather than blanking the whole check. Read that box for
  what specifically failed.
- The comparison table's **Model** column shows the names you gave when
  saving, not generic labels like `model_0`, and any plots below include a
  caption mapping those generic labels (baked into the images themselves) back
  to your names.
- If a ranking includes a Pareto-k reliability warning, it appears above the
  ranking table, with your model names substituted in.
- After **Use Selected Model**, a green message confirms which model is now
  active. Check Convergence Evaluation, Update Model, or SAE Estimation next
  for that model.

## If it fails

**Posterior predictive check shows an error for one or both plots.**
Each plot renders independently, so one failing doesn't block the other. The
error box names which plot failed and why.

**"Select a model" (after clicking Compare Selected with nothing checked).**
Check at least one model's box first.

**Comparing models fitted to different data.**
`compare_models()` rejects this with an error. Models must share the same
response and rows to produce a meaningful ranking.

**"No model selected" (when clicking Use Selected Model).**
Pick a model from the dropdown first.

## Related

For why models don't need to have converged to be saved or compared, see {doc}`../explanation/convergence`.

## Next

The model reproduces the observed data: continue to {doc}`09-prior-sensitivity`. The model does not: go back to {doc}`03-specify-model`.