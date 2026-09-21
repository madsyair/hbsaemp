# 4 · Modeling

With the model built in {doc}`03-data-preparation`, this is where
inference actually happens: sanity-check the prior, run the sampler, and
save the fit for later comparison — all still on the **Modeling** tab.

## Prior Predictive Check

Before spending time on MCMC, check what the model considers plausible
*before* it has seen `y` at all.

1. Leave **n_draws** at its default, **50**.
2. Click **Run Prior Predictive Check**.

   ```{figure} images/04-prior-run.png
   :alt: n_draws field and Run Prior Predictive Check button, with the success message after running
   :width: 700px

   **n_draws: 50**, and the result after clicking **Run Prior Predictive
   Check**.
   ```

3. Read the **Prior Summary** table — one row per coefficient (the
   intercept and the three predictor slopes), all on the **logit** scale
   since that's this model's link:

   ```{figure} images/04-prior-summary.png
   :alt: Prior Summary table with mean, sd, and 89% interval for each coefficient
   :width: 700px

   **Prior Summary**: intercept and three predictor coefficients.
   ```

   Every row's 89% interval straddles zero (`proporsi_R805I`, for
   instance, spans roughly −10 to +11), and the standard deviations are
   large (1.6 to nearly 7). That's a **weakly informative** prior by
   design: it doesn't presuppose whether any predictor pushes CHE up or
   down, or by how much — it leaves that entirely to the data in
   {doc}`05-evaluation`'s fit. An interval that straddled zero this widely
   would be a problem for a prior meant to encode strong domain knowledge;
   for a default, data-led prior like this one, it's expected.

4. Read the **Prior Predictive ECDF Plot** — the black line is the ECDF of
   the *observed* `y` values, and each thin blue line is the ECDF of one
   of the 50 datasets simulated purely from the prior (no data involved
   yet):

   ```{figure} images/04-prior-ecdf.png
   :alt: Prior Predictive ECDF plot comparing simulated y datasets against the observed y distribution
   :width: 700px

   **Prior Predictive ECDF Plot**: observed `y` (black) against 50
   prior-simulated datasets (blue).
   ```

**Does this prior pass?** Yes — the observed data's ECDF (black) sits
inside the spread of the prior-simulated ECDFs (blue), meaning the prior
isn't ruling out the real data as implausible before fitting even starts,
which is the one thing a prior predictive check is actually gate-keeping
against. What it also shows, honestly, is that the prior is *permissive*
well beyond the observed range: the simulated datasets extend out past
4,000 CHE households in a district, far beyond the observed maximum of
285. That's the trade-off of a deliberately wide, weakly informative
prior — it covers the plausible range along with a lot of implausible
extremes, rather than covering the plausible range tightly. That's an
acceptable place to start for this case study, not a reason to go back
and narrow the prior by hand; if it turns out to matter later, the
**Prior Sensitivity** check on the **Model Comparison** sub-tab
(see {doc}`05-evaluation`) is the tool that would actually confirm
whether this wide prior is quietly dominating the posterior for any
parameter. For now: proceed to **Fit Model**.

## Fit Model

5. Click **Fit Model** — this runs MCMC sampling using the settings from
   {doc}`03-data-preparation` (`draws` 2000, `tune` 2000, `chains` 4,
   `target_accept` 0.9).

   ```{figure} images/04-fit-model.png
   :alt: Fit Model button and "The MCMC sampling has completed" message
   :width: 700px

   Clicking **Fit Model**.
   ```

6. Wait for **"The MCMC sampling has completed."**

## Save Model

7. Leave **Model name** at its default, **Model 1**, and click **Save
   Model**.

   ```{figure} images/04-save-model.png
   :alt: Model name field set to Model 1 and the Save Model button
   :width: 700px

   Saving the fit as **Model 1**.
   ```

   This snapshot is what {doc}`05-evaluation`'s **Model Comparison**
   sub-tab reads from later — saving now costs nothing and keeps this fit
   available even if a later configuration turns out worse.

## Posterior Predictive Check

One last check before moving to formal diagnostics: does the fitted
model, now that it has seen the data, actually reproduce data that looks
like the real thing?

8. Click **Run Posterior Predictive Check**.

   ```{figure} images/04-posterior-run.png
   :alt: Run Posterior Predictive Check button and completion message
   :width: 700px

   Clicking **Run Posterior Predictive Check**.
   ```

9. Read the **Posterior Predictive Plot** — same ECDF style as the prior
   predictive plot back in the Prior Predictive Check step, but now built
   from the *fitted* model instead of the raw prior:

   ```{figure} images/04-posterior-plot.png
   :alt: Posterior Predictive Plot comparing simulated y datasets against the observed y distribution, now tightly matching
   :width: 700px

   **Posterior Predictive Plot**: observed `y` (black) against
   posterior-simulated datasets (blue).
   ```

   The contrast with the prior predictive plot is the point of doing
   both. Before fitting, the simulated data extended out past 4,000 — the
   prior's honest but exaggerated sense of what's possible. After
   fitting, the blue lines hug the observed black line closely across the
   entire range, and the x-axis itself only extends to about 350 — close
   to the real data's maximum of 285. That contraction is the data doing
   its job: fitting has pulled the wide prior down to something that
   actually resembles the CHE counts this case study is trying to
   explain.

10. Read the **Rootgram** — one interval per district (514 in total),
    comparing each district's observed `y` (black dot) against the
    model's posterior predictive interval for that specific district
    (blue bar):

    ```{figure} images/04-posterior-rootgram.png
    :alt: Rootgram showing posterior predictive credible intervals per district against observed values
    :width: 700px

    **Posterior Predictive Credible Intervals by Observation** (rootgram).
    ```

    Most black dots sit inside or right at the edge of their blue
    interval, tracking the same rises and falls across districts — the
    model isn't systematically over- or under-predicting in one direction
    across the board. A handful of districts (a few of the highest-`y`
    points, and one isolated point past index 400) sit above their
    interval's top edge — districts where something the three predictors
    don't capture is pushing CHE higher than the model expects. With 514
    districts and one credible interval each, a small number of misses
    like this is normal, not a red flag on its own; it's worth
    remembering these particular districts if {doc}`05-evaluation` later
    shows any pattern connected to them.

**Can this move on to checking convergence?** Yes. A posterior predictive
check is about whether the model's *story* is broadly consistent with the
data it's supposed to explain — and here it is, both in the aggregate
(the ECDF match) and per-district (the rootgram's general tracking).
That's a different question from whether the *sampler* actually explored
the posterior properly, which is exactly what {doc}`05-evaluation`'s
R-hat and ESS diagnostics check next — a model can look right here and
still have converged poorly, so this check clears the way to look at
convergence, rather than replacing it.

Move on to {doc}`05-evaluation`.