# 3 · Specify a model

State which column is the response, which carry auxiliary information, pick a
family, and build the model — without fitting it yet.

## Before you start

You need data loaded — see {doc}`01-load-data`. Everything on this page happens
on the **Modeling** tab's **Model Building** sub-tab, and none of it samples:
building a model here is as cheap as it is in the Python API, and mistakes are
free to fix.

## Do it

1. Pick a **Response Variable (y)** — the outcome you want to estimate.

   ```{figure} images/03-step1-response-variable.png
   :alt: Response Variable (y) selector
   :width: 600px

   The **Response Variable (y)** selector.
   ```

2. Check the boxes for your **Auxiliary / Predictor Variables (x)** — one
   checkbox per numeric column. Nothing is pre-checked: the app does not guess
   which columns are meant as predictors, since a wrong guess (picking up a
   simulation ground-truth column, or a column meant for a different family
   parameter) is easy to make automatically and hard to notice.

   ```{figure} images/03-step2-predictor-checkboxes.png
   :alt: Auxiliary / Predictor Variables (x) checkboxes
   :width: 600px

   The **Auxiliary / Predictor Variables (x)** checkboxes.
   ```

3. Optionally pick an **Area / Group Variable** — the column identifying each
   small area.

   ```{figure} images/03-step3-group-variable.png
   :alt: Area / Group Variable selector
   :width: 600px

   The **Area / Group Variable** selector.
   ```

4. Choose an **HB Family** from the dropdown. The description under it explains
   what that family expects:
   - **Gaussian** — continuous, unbounded response. Pass a sampling-variance
     column to enable the Fay-Herriot offset.
   - **Beta** — a proportion or rate.
   - **Binomial** — a count of successes out of a number of trials; the
     **Trials column (n_i)** becomes a required field once selected.

   ```{figure} images/03-step4-family-dropdown.png
   :alt: HB Family dropdown with description
   :width: 600px

   The **HB Family** dropdown, with its description underneath.
   ```

5. Fill in whichever family-specific fields appear below the family
   description (e.g. **Sampling variance column (D_i)** for Gaussian,
   **Trials column (n_i)** for Binomial).

   ```{figure} images/03-step5-family-specific-fields.png
   :alt: Family-specific fields such as Sampling variance column or Trials column
   :width: 600px

   Family-specific fields, e.g. **Sampling variance column (D_i)** for Gaussian.
   ```

6. **Optional — Pin a parameter.** Some families let you pin a distributional
   parameter instead of estimating it:
   - Check **Pin sigma** (Gaussian) or **Pin kappa** (Beta) to reveal a choice
     between **Column** (map it to an existing numeric column) or **Fixed
     value** (a single number for every area).
   - Pinning a parameter disables the field it would otherwise conflict with
     — pinning `sigma`, for example, disables **Sampling variance column**,
     since supplying both raises an error rather than silently picking one.

   ```{figure} images/03-step6-pin-parameter.png
   :alt: Pin sigma / Pin kappa checkbox with Column and Fixed value options
   :width: 600px

   The **Pin sigma**/**Pin kappa** checkbox, with the **Column**/**Fixed value**
   choice it reveals.
   ```

7. Adjust **Sampler Configuration** if you don't want the defaults (`draws`
   1000, `tune` 1000, `chains` 4, `cores` 1, `target_accept` 0.8). Check
   **Fix random seed** to make the run reproducible.

   ```{figure} images/03-step7-sampler-configuration.png
   :alt: Sampler Configuration fields and Fix random seed checkbox
   :width: 600px

   The **Sampler Configuration** fields and the **Fix random seed** checkbox.
   ```

8. Read the **Formula Preview** — it updates live as you change any of the
   above, and shows an error box instead of a formula if the current
   selection can't build a model yet (e.g. no response chosen).

   ```{figure} images/03-step8-formula-preview.png
   :alt: Formula Preview box
   :width: 600px

   The **Formula Preview** box.
   ```

9. Click **Build Model**.

   ```{figure} images/03-step9-build-model-button.png
   :alt: Build Model button
   :width: 600px

   The **Build Model** button.
   ```

## Check it worked

A green success message reads "Model built and data validated successfully.
Moving on to Prior Predictive Check." — sometimes followed by a note that some
rows were dropped for missing values in the columns you're using. Below that,
a **read-only Python code preview** shows the equivalent `hbsaemp` script for
this exact model, which you can save with **Save Code (.py)** to reproduce it
outside the GUI.

```{figure} images/03-code-preview-and-save-button.png
:alt: Read-only Python code preview with the Save Code (.py) button
:width: 600px

The read-only Python code preview, with the **Save Code (.py)** button.
```

If you don't see this, the **Formula Preview** box above the button should
already be telling you why — clicking **Build Model** re-checks the same
thing rather than surfacing a separate error.

## If it fails

**"Invalid formula."**
Something about the response/predictor/group selection can't form a valid
formula. Recheck your variable choices.

**"Family not available."**
The chosen family isn't registered the way the app expects. Try a different
family.

**"Data does not meet the family's requirements."**
For example, Binomial needs the trials column to be filled in and to make
sense against the response column (successes can't exceed trials). Fix the
data or the field mapping and click **Build Model** again.

**"Invalid configuration."**
A pinned value or sampler setting is out of range (e.g. `target_accept`
outside 0–1). Adjust it and retry.

## Related

For why an empty predictor list is the default rather than every remaining
numeric column, see {doc}`../explanation/index` (a page on this specific
choice lands with the remaining explanation pages).