# 3 · Data Preparation

CRISP-DM's Data Preparation stage usually covers three things: cleaning
the data, deciding which variables actually enter the model, and
transforming them into the model's expected shape.

**Selecting variables already happened.** The whole point of
{doc}`02-data-understanding`'s Scatter & Correlation step was exactly
this — narrowing 32 candidate PODES columns down to the three
(`proporsi_R704IK2`, `proporsi_R704JK2`, `proporsi_R805I`) with the
strongest, most defensible relationship to `est_prop_logit`. What's left
for this stage is entering that decision into the app and letting it
clean and validate the result — which happens on the **Modeling** tab's
**Model Building** sub-tab, not a separate "Data Preparation" screen.

1. Set **Response Variable (y)** to `y` — the raw count of CHE households
   per district, not `est_prop`. Keeping the response as a count (rather
   than a pre-computed proportion) matters for the family choice in a
   moment.

2. Set **Area / Group Variable** to `IDKABKOT` — the district code. This
   is what lets the model fit a random intercept per district rather than
   pooling all 514 districts into one flat regression.

3. Under **Auxiliary / Predictor Variables (x)**, check the three
   predictors chosen in {doc}`02-data-understanding`:
   `proporsi_R704IK2` (X1), `proporsi_R704JK2` (X2), and `proporsi_R805I`
   (X3). Leave everything else unchecked — in particular, `est_prop` and
   `rse_prop` are outputs of the direct estimation this case study is
   trying to improve on, not inputs to the model that replaces it.

   ```{figure} images/03-variables.png
   :alt: Response Variable, Area/Group Variable, and Predictor checkboxes set for the CHE model
   :width: 700px

   Response, group, and predictor selection.
   ```

4. Leave **Include intercept** checked. Without it, the model would be
   forced to assume the CHE rate is exactly zero when all three
   predictors are zero — an assumption with no basis here. The intercept
   lets the model fit its own baseline log-odds of CHE and estimate each
   predictor's effect relative to that baseline, rather than relative to
   an assumed-zero starting point.

5. Set **HB Family** to **binomial** and **Link Function** to **logit**.
   This is what makes the model "logit-normal": the linear predictor and
   the district random effect both live on the logit scale, exactly the
   scale checked against every predictor back in
   {doc}`02-data-understanding`'s Scatter & Correlation step. Binomial is
   the right family here — rather than, say, Beta on `est_prop` directly
   — because the raw data already gives the count behind the proportion
   (`y` successes out of `n` trials): a Binomial likelihood uses that
   count directly, so a district with 4,529 surveyed households
   contributes more precisely to the fit than one with 957, which fitting
   only `est_prop` as a continuous number would lose. Picking binomial is
   also a domain requirement, not just a label — it's what makes the next
   field required.

   ```{figure} images/03-family-link.png
   :alt: HB Family set to binomial and Link Function set to logit, with the required Trials column
   :width: 700px

   **HB Family**: binomial, **Link Function**: logit, with **Trials
   column (n_i)** now required.
   ```

6. Set **Trials column (n_i)** to `n`. Binomial requires it: it's what
   turns `y` from a bare count into "`y` successes out of `n` trials" —
   without it, the model has no way to know a count of 25 out of 2,101
   surveyed households means something different from 25 out of 500. This
   is exactly the kind of family/data mismatch this stage exists to catch
   early — before spending time on MCMC, not after.

7. In **Sampler Configuration**, use `draws` **2000**, `tune` **2000**,
   `chains` **4**, `cores` **1**, `target_accept` **0.9**:
   - `draws` and `tune` are both raised from the app's defaults (1000
     each) — a hierarchical model with 514 group-level effects needs more
     post-warmup draws than a simple regression to get stable R-hat and
     ESS in {doc}`05-evaluation`.
   - `chains` **4** is what makes R-hat computable at all — R-hat compares
     variance *between* chains to variance *within* them, so it needs
     more than one.
   - `target_accept` **0.9**, above the default 0.8, tells the sampler to
     take smaller steps. Hierarchical models like this one commonly
     produce a "funnel"-shaped posterior geometry around each district's
     random effect, and a higher target acceptance rate reduces the
     divergent transitions that geometry tends to cause.
   - `cores` **1** simply keeps this run single-threaded; raise it if your
     machine has cores to spare and you want the four chains to run in
     parallel instead of in sequence.

   ```{figure} images/03-sampler-config.png
   :alt: Sampler Configuration set to draws 2000, tune 2000, chains 4, cores 1, target_accept 0.9
   :width: 700px

   **Sampler Configuration**: draws 2000, tune 2000, chains 4, cores 1,
   target_accept 0.9.
   ```

8. Check **Fix random seed**, with `random_seed` **42**. This makes the
   sampling reproducible — running **Fit Model** later reproduces the
   same posterior draws, rather than a fresh random one each time.

9. Read the **Formula Preview**. As soon as steps 1–6 are filled in,
   `_update_preview()` fires automatically and tries to build a draft
   model right then — this is the first of two automatic checks this
   stage runs, and it happens live, before you click anything:
   - If the form isn't complete yet, the box shows a pending message
     instead of a formula ("select a response and at least one
     predictor").
   - If the current selection can't build a valid model at all, the box
     shows that error immediately.
   - Once buildable, it shows the actual formula. It should read exactly

     ```
     y ~ proporsi_R704IK2 + proporsi_R704JK2 + proporsi_R805I + (1|IDKABKOT)
     ```

     — the three predictors as fixed effects, plus `(1|IDKABKOT)` for the
     per-district random intercept. If yours doesn't match this, recheck
     steps 1–3 before continuing.

   ```{figure} images/03-formula-preview.png
   :alt: Formula Preview showing the completed formula, and the green success message after clicking Build Model
   :width: 700px

   **Formula Preview** once the form is complete, and the result of
   clicking **Build Model** below it.
   ```

10. Click **Build Model**. This re-runs `_update_preview()` once more to
    get a fresh draft, then — if that draft exists — calls the second
    automatic check, `check_data()`, on it: the same pre-flight and
    validation work `fit()` would do, minus the actual sampling, so a
    data problem surfaces now rather than after an expensive MCMC run.
    "No warning" here specifically means the columns, types, and value
    domains you picked all match what **binomial** requires — you should
    see "Model built and data validated successfully. Moving on to Prior
    Predictive Check." (see the figure above). If the draft from step 9
    didn't build, you'll instead see "Cannot build model: Fix the formula
    preview above first" — that's `_update_preview()` catching the
    problem, not `check_data()`.

    ```{note}
    The 24 districts with `rse_prop` missing aren't a cleaning problem to
    fix here. `rse_prop` isn't used as a model input — only `y`, `n`, and
    the three predictors are — so those 24 `NA` values simply never enter
    the model, and `check_data()` never has a reason to flag them.
    ```

11. Below the button, a read-only Python translation of everything just
    configured appears. Click **Save Code (.py)** to keep it — it's the
    exact `hbsaemp` script that reproduces this model outside the GUI.

    ```{figure} images/03-code-preview.png
    :alt: Read-only Python code preview with the Save Code (.py) button
    :width: 700px

    The generated `hbm_binomial(...)` script.
    ```

Move on to {doc}`04-modeling`, which picks up from here — Prior
Predictive Check, Fit Model, and Save Model.