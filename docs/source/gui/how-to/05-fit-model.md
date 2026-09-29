# 5 · Fit the model

Run the sampler and save the fit for later comparison. Both are on the **Modeling** tab, on the **Fit Model** sub-tab, after the **Prior Predictive Check**.

## Before you start

You need a built model whose prior you have checked. See {doc}`03-specify-model` and {doc}`04-check-priors`. Fitting is the one expensive step in this workflow: everything on the Model Building sub-tab is instant, and this is not.

## Do it

::::{tab-set}

:::{tab-item} 1. Fit Model
On the **Fit Model** sub-tab:

1. Click **Fit Model**.

   ```{figure} images/04-fit-model-button.png
   :alt: Fit Model button
   :width: 600px

   The **Fit Model** button.
   ```

2. A progress bar and an elapsed-time counter appear while MCMC sampling
   runs. There's no ETA, since runtime depends too much on model size and
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

:::{tab-item} 2. Save Model (optional)
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

Saving takes a snapshot for later use in **Results → Model Comparison**. It
does not check or require convergence (see {doc}`../explanation/convergence`
for why). You can fit again with different settings and save another named
snapshot without losing this one.
:::

::::

## Check it worked

- Fit: a green "The MCMC sampling has completed" message, and the **Save
  Model** controls become enabled.
- Save: a green "Saved as **\<name\>**." message.

## If it fails

**An error box during fitting (e.g. "Invalid prior specification").**
The message is passed through from the backend. Check the specific field it
names. This does not lose your configuration; adjust and click **Fit Model**
again.

**"No fitted model." (when clicking Save Model)**
Fit a model first. Save Model stays disabled until a fit completes.

**"Name required."**
The Model name field is empty. Give it a name before clicking Save Model.

## Related

For why saving doesn't check convergence, see {doc}`../explanation/convergence`.

## Next

Continue to {doc}`06-check-convergence`.