# 4 · Check the priors

Simulate data from the priors alone and see what the model considers plausible before it has seen any outcomes.
This is on the **Modeling** tab, on the **Prior Predictive Check** sub-tab, after **Model Building**.

## Before you start

You need a built model. See {doc}`03-specify-model`. The Prior Predictive Check is free, and it is your last chance to catch an implausible prior before spending the time to sample.

## Do it

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

This doesn't touch your data's likelihood at all. It only tells you what the
model considers plausible *before* seeing any outcomes.

## Does the prior pass?

Compare the observed data (black) with the simulated datasets (blue) on the **Prior Predictive ECDF Plot**.

- The prior passes when the observed data's ECDF sits inside the spread of the simulated ECDFs. The prior does not rule out the real data before fitting starts.
- A very wide spread beyond the observed range is expected for a deliberately wide, weakly informative prior. See {doc}`../tutorials/case-study-che/04-modeling` for a worked example.
- The prior fails when the observed data falls outside the spread of the simulated data. Go back to {doc}`03-specify-model`, change the model, click **Build Model**, and run the check again.

## Check it worked

- Prior check: the ECDF plot renders without an error box, and the summary
  table has one row per parameter.

## If it fails

**"Cannot run prior predictive check" / "Fix the formula preview above
first."**
The model draft is invalid. Go back to {doc}`03-specify-model` and get a
successful **Build Model** first.

## Next

Prior passes: {doc}`05-fit-model`. Prior does not pass: {doc}`03-specify-model`.