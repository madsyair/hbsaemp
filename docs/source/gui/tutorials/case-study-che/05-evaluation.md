# 5 · Evaluation

Model 1 fit without errors and passed its posterior predictive check in
{doc}`04-modeling` — but neither of those confirms the *sampler* actually
explored the posterior properly. That's what this stage checks first,
before trusting anything downstream.

## Convergence Evaluation

1. Open **Results → MCMC Convergence Evaluation**.
2. Click **Load Convergence Diagnostics**.

   ```{figure} images/05-load-diagnostics.png
   :alt: Load Convergence Diagnostics button and "Diagnostics computed" message
   :width: 700px

   Clicking **Load Convergence Diagnostics**.
   ```

3. Read the **R-hat and ESS** table. The description above it states the
   thresholds this case study will use: **R-hat ≤ 1.01**, **ESS ≥ 400**
   (100 × the 4 chains configured in {doc}`03-data-preparation`).

   ```{figure} images/05-rhat-ess-table.png
   :alt: R-hat and ESS table for Model 1, showing two predictors exceeding the R-hat threshold
   :width: 700px

   **R-hat and ESS** for Model 1 — page 1 of 5 (~518 parameters: 5
   population-level terms plus one random intercept per district).
   ```

**Reading the table.** The population-level terms split into two groups:

| Parameter | r_hat | ess_bulk | Within threshold? |
|---|---|---|---|
| `Intercept` | 1.009 | 615 | Yes |
| `proporsi_R704IK2` (X1) | **1.019** | **323** | **No** — both r_hat and ESS fail |
| `proporsi_R704JK2` (X2) | **1.017** | **352** | **No** — both r_hat and ESS fail |
| `proporsi_R805I` (X3) | 1.009 | 457 | Yes, at the edge |
| `1|IDKABKOT_sigma` | 1.005 | 753 | Yes |

The per-district random intercepts shown on this first page
(`1|IDKABKOT[1101]` through `[1110]`) all sit at r_hat 1.000–1.009 with
ESS in the thousands — the group-level structure itself is sampling
fine. The problem is narrow and specific: **X1 and X2's population-level
slopes** are the only two parameters (out of roughly 518) failing on
*both* metrics at once, not just sitting near the threshold on one.

**Can this move on as-is, or does it need a refit?** A refit is the right
call here — not because the app requires it (see
{doc}`../../explanation/convergence` for why the app never blocks you
either way), but because two coefficients failing r_hat *and* ESS
together is a genuine mixing problem for exactly the two predictors this
case study picked as its strongest candidates back in
{doc}`02-data-understanding`. Trusting Model 1's estimates for X1 and X2
before they've mixed properly would undermine the reason those two
variables were chosen in the first place.

## Update Model

Move to **Results → Update Model** to refit with adjusted sampler
settings, without going back to {doc}`03-data-preparation` to rebuild the
model from scratch — same formula, same family, just a harder-working
sampler.

4. Check **Override draws** and set it to **4000** (double the original
   2000) — more post-warmup samples directly raises ESS for the two
   parameters that fell short.
5. Check **Override target_accept** and set it to **0.95** (up from 0.9)
   — a still-smaller step size, one more push against the same
   funnel-shaped geometry {doc}`03-data-preparation` already flagged as
   the likely cause. Leave every other override unchecked: `tune`,
   `chains`, `cores`, and `random_seed` stay exactly as they were.

   ```{figure} images/05-update-sampler-overrides.png
   :alt: Sampler Overrides with draws set to 4000 and target_accept set to 0.95
   :width: 700px

   **Override draws**: 4000, **Override target_accept**: 0.95 — everything
   else unchecked.
   ```

6. Click **Update Model** and wait for the refit to finish.

   ```{figure} images/05-update-model-success.png
   :alt: Update Model button and success message after refitting
   :width: 700px

   "Model refit complete."
   ```

7. Confirm the new configuration in the generated summary: `draws: 4000`,
   `tune: 2000`, `chains: 4`, `target_accept: 0.95`, same formula, same
   514 rows — only the two overridden settings changed.

   ```{figure} images/05-update-formula-config.png
   :alt: Updated model configuration summary showing draws 4000 and target_accept 0.95
   :width: 700px

   Confirming the refit picked up both overrides.
   ```

8. Worth a quick re-check before trusting the new fit: re-run **Posterior
   Predictive Check** back on the Modeling tab. Both plots still look the
   same as {doc}`04-modeling`'s originals — the model's fit to the data
   hasn't changed, only how thoroughly the sampler explored it has.

   ```{figure} images/05-update-posterior-plot.png
   :alt: Posterior Predictive Plot after the refit, matching the original
   :width: 600px

   Posterior Predictive Plot after refitting — unchanged from before.
   ```

   ```{figure} images/05-update-posterior-rootgram.png
   :alt: Rootgram after the refit, matching the original
   :width: 600px

   Rootgram after refitting — unchanged from before.
   ```

9. Back on **Results → MCMC Convergence Evaluation**, click **Load
   Convergence Diagnostics** again.

   ```{figure} images/05-update-rhat-ess.png
   :alt: R-hat and ESS table after the refit, showing all parameters now within threshold
   :width: 700px

   **R-hat and ESS** after the refit.
   ```

**Is it converged now?** Yes — every population-level parameter clears
both thresholds, most by a wide margin:

| Parameter | r_hat before → after | ess_bulk before → after |
|---|---|---|
| `Intercept` | 1.009 → 1.002 | 615 → 1,607 |
| `proporsi_R704IK2` (X1) | 1.019 → **1.003** | 323 → **815** |
| `proporsi_R704JK2` (X2) | 1.017 → **1.007** | 352 → **668** |
| `proporsi_R805I` (X3) | 1.009 → 1.003 | 457 → 812 |
| `1|IDKABKOT_sigma` | 1.005 → 1.010 | 753 → 906 |

The two parameters that failed before — X1 and X2 — are now comfortably
under 1.01, with ESS more than double the 400 threshold. `1|IDKABKOT_sigma`
sits exactly at 1.01, the edge of the threshold rather than clearly inside
it; combined with its ESS also having improved (753 to 906), that's a
pass, not a lingering concern.

```{note}
Doubling `draws` and raising `target_accept` both worked in the same
direction here, so this case study can't tell you which one mattered
more — only that the combination was enough. If a future model needs
this same fix and either change alone is expensive (more draws costs
runtime, higher target_accept costs it too), it's worth trying them one
at a time to see which is doing the real work.
```

This refit is worth keeping. Go back to the **Modeling** tab and **Save
Model** again — name it **Model 2** rather than reusing "Model 1", so
both the original and the refit are available side by side on
**Model Comparison** next, instead of the refit silently replacing the
evidence that the fix was needed at all.

## Save the refit as Model 2

10. On the **Modeling** tab's **Fit Model** sub-tab, set **Model name** to
   `Model 2` and click **Save Model**.

   ```{figure} images/05-save-model2.png
   :alt: Model name field and Save Model button, with "Saved as Model 2" success message
   :width: 700px

   "Saved as **Model 2**." (The field itself has already reset to suggest
   `Model 3` as the next default name — that's just a suggestion for a
   future save, not a sign anything went wrong.)
   ```

## Confirm convergence beyond the summary table

The R-hat/ESS table already showed every parameter clearing threshold —
worth also glancing at the plots below that table, which show *how* the
sampler behaved, not just the two summary numbers.

```{figure} images/05-trace-density.png
:alt: Trace plot and density plot for Model 2's five main parameters
:width: 700px

**Trace Plot** and **Density Plot**.
```

The **Trace Plot** shows all four chains (four colors) for each
parameter overlapping into one dense, uniform band with no chain
drifting off on its own — the "fuzzy caterpillar" look that indicates
good mixing. The **Density Plot** underneath shows a single smooth,
unimodal peak per parameter, consistent with the means already reported
in the table (X1 ≈ 1.7, X2 ≈ 0.51, X3 ≈ 1.7, Intercept ≈ −5.2).

```{figure} images/05-autocorr-rhatdist.png
:alt: Autocorrelation plot and R-hat distribution plot across all parameters
:width: 700px

**Autocorrelation Plot** and **R-hat Distribution Plot**.
```

The **Autocorrelation Plot** drops toward zero within about 20–40 lags
for every parameter and stays inside the shaded band afterward — draws
decorrelate quickly, which is exactly what high ESS numbers require. The
**R-hat Distribution Plot** is the most reassuring of the six: it's a
cumulative view across *every* parameter in the model, not just the five
shown in the earlier table — including all 514 district random
intercepts — and the curve reaches 1.0 well before the dashed line at
1.01. Every parameter in the model clears the threshold, not only the
handful visible on the table's first page.

```{figure} images/05-ess-nuts.png
:alt: Effective sample size plot and NUTS energy/BFMI plot
:width: 700px

**Effective Sample Size Plot** and **NUTS Energy / BFMI Plot**.
```

The **Effective Sample Size Plot** shows local ESS across quantiles
(not just the bulk/tail summary numbers) staying between roughly 5,000
and 15,000 for every parameter — far above the 400 threshold at every
point in the distribution, not only on average. The **NUTS Energy/BFMI
Plot** shows all four chains' BFMI values (left) comfortably above the
0.3 reference line, and the marginal and transition energy distributions
(right) overlapping closely in shape — no sign of the sampler getting
stuck in high-energy regions. Combined with the R-hat/ESS table from
before, **Model 2 has converged** on every diagnostic this page checks,
not just the two summary numbers.

## Model Comparison

With both a non-converged and a converged fit saved, this is a natural
point to compare them — not just on fit quality, but on whether the
convergence problem in Model 1 changed any substantive conclusion.

11. Open **Results → Model Comparison**.
12. Check both **Model 1** and **Model 2**.
13. Check **Include Bayes Factor (Savage-Dickey, per coefficient)** — to
   see whether each predictor's evidence for being non-zero holds up the
   same way regardless of which fit is used.
14. Check **Include Prior Sensitivity (power-scaling, per parameter)** —
   to finally settle the question {doc}`04-modeling`'s Prior Predictive
   Check left open: does the wide, weakly informative prior end up
   quietly dominating any parameter's posterior, now that there's an
   actual fit to check it against?
15. Click **Compare Selected**.

   ```{figure} images/05-model-comparison-setup.png
   :alt: Model Comparison with Model 1 and Model 2 checked, Bayes Factor and Prior Sensitivity enabled
   :width: 700px

   **Model 1** and **Model 2** checked, both extra diagnostics enabled.
   ```

### Ranking (LOO / ELPD)

```{figure} images/05-ranking-loo.png
:alt: Ranking table by LOO/ELPD for Model 1 and Model 2
:width: 700px

**Ranking (LOO / ELPD)**.
```

Model 1 ranks first (`elpd_diff` 0.0, the reference), with Model 2
scoring 26.0 lower — about 6.5 times its own standard error (`dse` 4.0),
and `p_worse` 1.0 suggesting that difference is close to certain under
this metric. Taken at face value, that says Model 1 predicts held-out
data better than Model 2.

```{important}
Taken at face value is doing a lot of work in that sentence. The
`diag_elpd` column reads **453 k̂ > 0.70** for Model 1 and **491 k̂ >
0.70** for Model 2 — out of 514 districts, the overwhelming majority of
both models' per-district PSIS-LOO estimates are flagged unreliable.
When almost every point estimate behind a ranking is untrustworthy, the
ranking built from them is untrustworthy too. This isn't a reason to
prefer Model 1 over Model 2 on ELPD grounds — it's a reason not to trust
ELPD much for either model here, likely a consequence of how variable
`n` is across districts (957 to 4,529) interacting with PSIS-LOO's
importance-sampling assumptions. Read the Bayes Factor and Prior
Sensitivity results below instead of leaning on this ranking.
```

### Bayes Factor

```{figure} images/05-bayes-factor.png
:alt: Savage-Dickey Bayes Factor per coefficient for Model 1 and Model 2
:width: 700px

**Bayes Factor** (Savage-Dickey, BF10 > 1 favors including the term).
```

| Term | Model 1 BF10 | Model 2 BF10 | Strength |
|---|---|---|---|
| `proporsi_R704IK2` (X1) | 11.1 | 11.8 | Strong evidence for inclusion, both models agree |
| `proporsi_R704JK2` (X2) | 5.2 | 7.0 | Moderate evidence for inclusion, both models agree |
| `proporsi_R805I` (X3) | 23.5 | 11.7 | Strong evidence for inclusion, both models agree |

Every coefficient clears BF10 > 1 in both models, by a similar order of
magnitude despite Model 1's convergence problem. That's the answer to
the question this comparison was really asking: **the mixing problem in
Model 1 didn't change which predictors matter.** X1 and X3 have strong
support, X2 has moderate support, consistently, whether or not the
sampler had fully converged.

### Prior Sensitivity

```{figure} images/05-prior-sensitivity.png
:alt: Prior Sensitivity power-scaling table for the Intercept, Model 1 and Model 2
:width: 700px

**Prior Sensitivity** — Intercept shown; page 1 of 5.
```

For the Intercept, **likelihood sensitivity clearly exceeds prior
sensitivity** in both models (Model 1: 0.249 vs. 0.027; Model 2: 0.363
vs. 0.025) — roughly nine to fifteen times larger. By the metric
explained above the table, that means scaling the *data's* influence
moves this parameter's posterior far more than scaling the *prior's*
influence does: the Intercept is data-driven, not prior-driven, in both
fits. That's the direct answer to the concern {doc}`04-modeling` raised
about the prior's very wide 89% intervals — at least for the Intercept,
the wide prior isn't quietly steering the result. Checking the same
pattern holds for X1, X2, and X3 means paging through the remaining four
pages of this table; the Intercept alone doesn't confirm all four
coefficients, only that the same wide-prior concern didn't materialize
for this one.

Move on to {doc}`06-deployment`.