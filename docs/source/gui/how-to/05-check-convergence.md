# 5 · Check convergence

Read the R-hat/ESS table and diagnostic plots for the model currently in use.
The app shows these numbers as-is — deciding whether a fit is trustworthy is
left to you (see {doc}`../explanation/convergence`).

## Before you start

You need a fitted model — see {doc}`04-fit-model`. This page covers the
**Results** tab's **MCMC Convergence Evaluation** sub-tab.

## Do it

1. Open **Results → MCMC Convergence Evaluation**.
2. Click **Load Convergence Diagnostics**.
3. Open the **R-hat and ESS** accordion to see the per-parameter table.
4. Scroll down to the diagnostic plots (trace, density, autocorrelation,
   R-hat distribution, effective sample size, NUTS energy/BFMI) — only the
   ones the backend actually produced are shown.
5. Optionally click **Download Plots (PDF)** to save all rendered plots as one
   file.

## Check it worked

- The description above the table states the commonly used thresholds
  (**R-hat ≤ 1.01**, **ESS ≥** 100 × the number of chains) — read these
  against your own study's requirements, not as a pass/fail the app enforces.
- The table has one row per parameter, with `r_hat`, `ess_bulk`, and
  `ess_tail` columns rounded for display (the underlying numbers used for any
  internal comparison are full precision).
- A green "Diagnostics computed." message confirms the run finished; it does
  not mean the fit converged, only that the diagnostics were successfully
  computed.

## If it fails

**"Model has not been fitted." / "Please fit the model first."**
Go back to {doc}`04-fit-model` and complete a fit before loading diagnostics.

**Plots are missing for some diagnostic types.**
Not every plot type is always produced by the backend — the page only shows
what's actually available; nothing is broken if fewer than six appear.

**"Download Plots (PDF)" is disabled.**
No plots were generated for this fit yet — load diagnostics first.

## Related

For why the app doesn't block Save Model or Model Comparison on these
numbers, see {doc}`../explanation/convergence`.