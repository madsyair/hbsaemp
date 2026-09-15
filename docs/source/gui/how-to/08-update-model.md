# 8 · Update a model

Refit the model currently in use with different sampler settings, a modified
formula, or replacement data — without rebuilding it from scratch on the
Modeling tab.

## Before you start

You need a fitted model — see {doc}`04-fit-model`. This page covers the
**Results** tab's **Update Model** sub-tab. `update_model()` is a **full
refit**, not a resumption: whatever you don't override stays the same as the
current model.

## Do it

1. Open **Results → Update Model**. The **Current Fitted Model** card
   summarizes what's active now.
2. **Optional — override sampler settings.** Check **Override draws**,
   **Override tune**, **Override chains**, **Override cores**, **Override
   target_accept**, **Override max_treedepth**, or **Override random_seed**
   individually — each checkbox enables its own input field. Anything left
   unchecked keeps the current model's value.
3. **Optional — update the formula.** In **Formula Update**, type a template
   like `. ~ . + x3 - x1` to add `x3` and remove `x1` while keeping everything
   else the same. A live preview shows the resulting formula string; whether
   the new column actually exists in the data is only checked when you click
   **Update Model**, not here.
4. **Optional — refit on different data.** Check "Refit on different data"
   and either upload a replacement CSV or pick a built-in dataset. Formula,
   family, link, and group stay the same unless you also changed the formula
   above.
5. Click **Update Model**.
6. Watch the progress bar and elapsed-time counter while the refit runs.

## Check it worked

- A green "Model refit complete" message appears, pointing you to **Save
  Model** on the Modeling tab if you want to keep this version, or back to
  the other Results sub-tabs to see updated diagnostics and estimates.
- If the backend had to quietly fix something in your replacement data (for
  example, copying over a missing design column when the row count matches),
  that's reported as a note appended to the success message — not hidden.
- Every other Results sub-tab (Convergence Evaluation, SAE Estimation) resets
  to "Model changed — run again to refresh the results" the moment the
  update completes, since their old numbers described the previous fit.

## If it fails

**"Model has not been fitted."**
There's no active model to update — fit one first on the Modeling tab.

**"Invalid formula" (from the Formula Update field).**
The template couldn't be applied to the current formula — check the syntax
(`. ~ . + newcol - oldcol`).

**"Replacement data does not meet the family's requirements."**
The new data is missing a column the family needs, or it doesn't validate
the same way the original data did.

**"Invalid configuration."**
An overridden sampler value is out of range.

## Related

For why Update Model lives here inside Results instead of as its own
top-level tab, see {doc}`../explanation/results`.