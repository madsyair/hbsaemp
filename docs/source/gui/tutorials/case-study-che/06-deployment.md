# 6 · Deployment

## Choosing which model to deploy

{doc}`05-evaluation`'s Model Comparison left three pieces of evidence on
the table, without picking a winner outright. Put together, they support
using **Model 1**, the original fit, for the estimates this stage hands
over to a policy audience:

- **Ranking (LOO / ELPD)** ranked Model 1 first — the only quantitative
  predictive-fit signal available, even with its Pareto-k reliability
  caveat noted.
- **Bayes Factor** reached the *same* conclusion in both models: all
  three predictors clear BF10 > 1, at a similar order of magnitude. The
  refit in Model 2 didn't change which variables matter or by roughly how
  much.
- Model 1's shortfall in {doc}`05-evaluation` was narrow and modest — two
  coefficients (X1, X2) with r_hat just over 1.01 (1.019 and 1.017) and
  ESS a bit under 400 (323 and 352), not an order-of-magnitude failure.
  Model 2 existed specifically to test whether fixing that changed
  anything substantive, and it didn't.

Taken together: the mixing problem was real enough to be worth fixing and
checking, which {doc}`05-evaluation` did — but once checked, it turned
out not to change any conclusion this case study relies on. Given that,
and given Model 1 is the one the ranking (for what it's worth) favors,
Model 1 is the model this stage carries forward. This is a judgment call
built on the comparison evidence, not a rule the app enforces either way
— see {doc}`../../explanation/convergence` for why.

1. Open **Results → Model Comparison** (or wherever Model 1 and Model 2
   are still listed from {doc}`05-evaluation`).
2. Set **Use this model** to **Model 1** and click **Use Selected
   Model**.

   ```{figure} images/06-use-selected-model.png
   :alt: Use this model dropdown set to Model 1, with confirmation message
   :width: 700px

   "**Model 1** is now the active model."
   ```

## Run SAE Estimation

3. Open **Results → SAE Estimation**.
4. Leave **Credible interval (ci_prob)** at its default, **0.95**. This
   case study isn't estimating for districts outside the training data,
   so leave "Estimate for new/unsampled areas" unchecked.
5. Click **Run SAE Estimation**.

   ```{figure} images/06-sae-run.png
   :alt: SAE Estimation settings and the Run SAE Estimation button, with the completion message
   :width: 700px

   "SAE estimation complete. Mean RSE: **17.10%**, Mean MSE: **0.0000**."
   ```

6. Review the results table, then click **Download CSV of SAE Results**
   to keep the full set of 514 estimates.

   ```{figure} images/06-sae-results-table.png
   :alt: SAE Estimation results table with one row per district
   :width: 700px

   Results table — one row per district (`mean`, `sd`, `ci_lower`,
   `ci_upper`, `rse_pct`, `mse`, `rmse`).
   ```

{download}`Download sae_estimation_model1.csv <data/sae_estimation_model1.csv>`

## Direct estimation vs. HB-SAE logit-normal

This is the comparison the whole case study has been building toward —
does small-area estimation actually solve the problem
{doc}`01-business-understanding` opened with?

| | min | mean | max | areas with RSE ≤ 25% |
|---|---|---|---|---|
| **Direct estimation** | 13.7% | **40.6%** | 100.3% | 114 of 490 with a defined RSE (24 more are undefined entirely) |
| **HB-SAE logit-normal** | 5.5% | **17.1%** | 50.5% | **427 of 514** |

Every figure moves the same direction. Mean RSE drops from 40.6% to
17.1% — well under half. The worst-case district goes from 100.3% (an
estimate as uncertain as the number itself) down to 50.5%. And the count
of districts precise enough to trust by the common 25% RSE benchmark goes
from 114 — under a quarter of districts, and only among the 490 with a
defined RSE at all — to **427 of all 514 districts**, RSE defined
everywhere because the model borrows strength from the auxiliary
predictors rather than relying on each district's own, often tiny,
survey sample.

```{note}
The 24 districts with undefined direct-estimation RSE (`est_prop = 0`,
flagged back in {doc}`02-data-understanding`) aren't a special case for
HB-SAE — every one of them gets a defined, finite RSE in the 514-row
result above, precisely because the model's estimate for a district
comes from its predictors and the shared random-effect structure, not
solely from that district's own zero-count sample.
```

Direct estimation puts the average CHE proportion at 0.0269 — in other
words, based on direct estimation, an average of 2.69% of the population
in each district experiences CHE. The logit-normal model run through the
`hbsaemp` package puts the same average at 0.0255 — based on indirect
estimation via the SAE HB logit-normal model, an average of 2.55% of the
population in each district experiences CHE.

The two numbers sit close together — SAE isn't shifting the overall
national picture of CHE, it's making that picture far more precise at
the level of each individual district. That's exactly what's expected:
SAE isn't a tool for changing the average, it's a tool for stabilizing
each area's individual estimate without distorting the overall pattern.

This is the concrete answer to {doc}`01-business-understanding`'s
opening problem: a district government can now get a CHE estimate with a
real, usable margin of error for its own area — something direct
estimation could not deliver for the majority of Indonesia's 514
districts.

This shows that the small-area estimation approach using the HB
logit-normal method produces more reliable estimates at the small-area
level — not just as a claim, but demonstrated by the numbers themselves:
mean RSE dropping from 40.6% to 17.1%, and the count of districts with a
usable estimate (RSE ≤ 25%) rising from 114 to 427 out of 514.