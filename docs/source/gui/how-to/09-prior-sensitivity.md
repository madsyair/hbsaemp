# 9 · Check prior sensitivity

Check whether a parameter's posterior is driven mainly by your prior rather than by the data. This is on **Results → Model Comparison**, next to the comparison you ran in {doc}`08-check-fit-and-compare`.

## Before you start

You need at least one model saved. See the **Save Model** step in {doc}`05-fit-model`. Prior Sensitivity costs an extra sampling pass per model on top of plain LOO, so it is off by default.

## Do it

1. On **Results → Model Comparison**, check the box next to each model you want to test, as in {doc}`08-check-fit-and-compare`.

2. Check **Include Prior Sensitivity (power-scaling, per parameter)**.

   ```{figure} images/06-step3b-prior-sensitivity-checkbox.png
   :alt: Include Prior Sensitivity (power-scaling, per parameter) checkbox
   :width: 600px

   The **Include Prior Sensitivity (power-scaling, per parameter)** checkbox.
   ```

3. Click **Compare Selected**.

4. Read the **Prior Sensitivity** table, with one column each for **prior**, **likelihood**, and **diagnosis** per parameter.

   ```{figure} images/06-step5b-prior-sensitivity-table.png
   :alt: Prior Sensitivity table with prior, likelihood, and diagnosis columns per parameter
   :width: 600px

   The **Prior Sensitivity** table, one column each for prior, likelihood,
   and diagnosis, per parameter.
   ```

## Read the table

A higher **prior** or **likelihood** value for a parameter means its posterior shifts more when that side is scaled up or down. The **diagnosis** column flags a parameter when the value is above 0.05.

| Diagnosis | When it appears | What it means |
| --- | --- | --- |
| ✓ | Neither flag below | No sensitivity issue flagged. |
| potential strong prior / weak likelihood | **prior** above 0.05 and **likelihood** below 0.05 | The posterior is driven mainly by your prior choice rather than by the data. |
| potential prior-data conflict | **prior** and **likelihood** both above 0.05 | Both the prior and the data move the posterior. They may disagree. |

Either flag is worth a second look before trusting that parameter's estimate.

## Check it worked

- If **Prior Sensitivity** is enabled, a higher **prior** or **likelihood**
  value for a parameter means its posterior shifts more when that side is
  scaled up or down. Watch the **diagnosis** column specifically. A note
  like "potential strong prior / weak likelihood" flags a parameter whose
  posterior is driven mainly by your prior choice rather than by the data,
  worth a second look before trusting that parameter's estimate.

## Next

No flags, or flags you can justify: continue to {doc}`10-estimate-areas`. Flags you cannot justify: go back to {doc}`03-specify-model`.
