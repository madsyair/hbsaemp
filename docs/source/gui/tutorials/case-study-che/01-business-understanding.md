# 1 · Business Understanding

A household faces *Catastrophic Health Expenditure* (CHE) when its direct
health spending exceeds 10% of its total household spending. CHE is not
just a budget problem — the WHO and Indonesia's SDG framework (target 3.8,
indicator 3.8.2) treat it as a marker that a household is being pushed
toward poverty by the cost of staying healthy.

## The problem

A household coping with a health shock by cutting food, or a low-income
household whose health costs already dominate its budget, is exactly the
population target 3.8's Universal Health Coverage goal is meant to protect.
Nationally, Indonesia's 2018 CHE rate was about **2.7% of the population**
— small as a share, but uneven across the country, which is why the policy
question is not "what is the national rate" but "which districts need
attention."

## Why district level, and why it's hard

A district government deciding where to expand health-financing protection
needs a CHE rate **for its own district** — a national or provincial figure
doesn't tell a *kabupaten/kota* whether it has a problem. But CHE is
measured through a household expenditure survey (SUSENAS) sized for
national and provincial precision, not district precision. Push the same
sample down to district level and the numbers get shaky:

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-text-center

:::{grid-item-card} 514
:class-card: sd-border-primary
districts analyzed, median 2,245 surveyed households per district
:::

:::{grid-item-card} ~40.6%
:class-card: sd-border-danger
average **RSE** from direct estimation — a common benchmark treats
anything past 25% as unreliable
:::

:::{grid-item-card} 24
:class-card: sd-border-danger
districts with zero recorded CHE households in sample — their RSE is
undefined, not merely large
:::

::::

```{important}
A number with no defined error margin isn't a weak estimate — it isn't
usable as one at all.
```

## Why small-area estimation

The direct fix — survey more households per district — is expensive at
national scale and slow to fund. Small-area estimation (SAE) takes the
other route: it borrows strength from *auxiliary* data that already exists
for every district (here, village census variables from PODES) to
stabilize the noisy direct estimates, without waiting for a bigger survey.

That trade — some model-based assumptions, in exchange for a usable number
in all 514 districts instead of a handful — is the reason this case study
exists, and the reason the rest of it walks through the app's SAE workflow
rather than reporting the direct estimates as final.