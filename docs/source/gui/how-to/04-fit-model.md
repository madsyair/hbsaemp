# 4 · Fit the model

Sanity-check the prior, run the sampler, save the fit for later comparison, and
check the posterior predictive fit — all on the **Modeling** tab, after
**Model Building**.

## Before you start

You need a built model — see {doc}`03-specify-model`. Fitting is the one
expensive step in this workflow: everything on the Model Building sub-tab is
instant, and this is not. The Prior Predictive Check below is free and worth
doing first, since it's your last chance to catch an implausible prior before
spending the time to sample.

## Do it

::::{tab-set}

:::{tab-item} 1. Prior Predictive Check (optional but free)
On the **Prior Predictive Check** sub-tab:

1. Adjust **n_draws** if you want more or fewer prior draws (default 50).

   ```{figure} images/04-prior-ndraws-field.png
   :alt: n_draws field for the Prior Predictive Check
   :width: 600px

   The **n_draws** field.
   ```

2. Click **Run Prior Predictive Check**.

   ```{figure} images/04-prior-run-button.png
   :alt: Run Prior Predictive Check button
   :width: 600px

   The **Run Prior Predictive Check** button.
   ```

3. Read the **Prior Summary** table and the **Prior Predictive ECDF Plot**.

   ```{figure} images/04-prior-summary-and-ecdf.png
   :alt: Prior Summary table and Prior Predictive ECDF Plot
   :width: 600px

   The **Prior Summary** table and the **Prior Predictive ECDF Plot**.
   ```

This doesn't touch your data's likelihood at all — it only tells you what the
model considers plausible *before* seeing any outcomes.
:::

:::{tab-item} 2. Fit Model
On the **Fit Model** sub-tab:

1. Click **Fit Model**.

   ```{figure} images/04-fit-model-button.png
   :alt: Fit Model button
   :width: 600px

   The **Fit Model** button.
   ```

2. A progress bar and an elapsed-time counter appear while MCMC sampling
   runs — there's no ETA, since runtime depends too much on model size and
   hardware to estimate reliably, but the counter confirms the app hasn't
   frozen.

   ```{figure} images/04-fit-progress-bar.png
   :alt: Progress bar and elapsed-time counter during MCMC sampling
   :width: 600px

   The progress bar and elapsed-time counter while sampling runs.
   ```

3. Wait for "The MCMC sampling has completed."

   ```{figure} images/04-fit-completed-message.png
   :alt: The MCMC sampling has completed success message
   :width: 600px

   The "The MCMC sampling has completed" message.
   ```
:::

:::{tab-item} 3. Save Model (optional)
Still on **Fit Model**, once a fit completes:

1. A **Model name** field is enabled, pre-filled with a default like
   `Model 1`. Rename it if you want something more memorable.

   ```{figure} images/04-save-model-name-field.png
   :alt: Model name field, pre-filled with a default name
   :width: 600px

   The **Model name** field, pre-filled with a default like `Model 1`.
   ```

2. Click **Save Model**.

   ```{figure} images/04-save-model-button.png
   :alt: Save Model button
   :width: 600px

   The **Save Model** button.
   ```

Saving takes a snapshot for later use in **Results → Model Comparison** — it
does not check or require convergence (see {doc}`../explanation/convergence`
for why). You can fit again with different settings and save another named
snapshot without losing this one.
:::

:::{tab-item} 4. Posterior Predictive Check (optional)
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

::::

## Check it worked

- Prior check: the ECDF plot renders without an error box, and the summary
  table has one row per parameter.
- Fit: a green "The MCMC sampling has completed" message, and the **Save
  Model** controls become enabled.
- Save: a green "Saved as **\<name\>**." message.
- Posterior check: both plots render. If only one does, the other shows its
  own error box rather than blanking the whole check — read that box for
  what specifically failed.

## If it fails

**"Cannot run prior predictive check" / "Fix the formula preview above
first."**
The model draft is invalid — go back to {doc}`03-specify-model` and get a
successful **Build Model** first.

**An error box during fitting (e.g. "Invalid prior specification").**
The message is passed through from the backend — check the specific field it
names. This does not lose your configuration; adjust and click **Fit Model**
again.

**"No fitted model." (when clicking Save Model)**
Fit a model first — Save Model stays disabled until a fit completes.

**"Name required."**
The Model name field is empty. Give it a name before clicking Save Model.

**Posterior predictive check shows an error for one or both plots.**
Each plot renders independently, so one failing doesn't block the other. The
error box names which plot failed and why.

## Related

For why saving doesn't check convergence, see {doc}`../explanation/convergence`.