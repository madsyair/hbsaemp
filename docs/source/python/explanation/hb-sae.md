# What hierarchical Bayes small area estimation is

Why a direct estimate is unreliable for a small area, and what borrowing strength does
about it.

## The question

A survey is designed to produce reliable national figures. Break the same sample down by
district and each district keeps only a handful of respondents. The estimate for that
district is still *unbiased* — it is the right calculation — but its variance is so
large that the number is useless for making a decision.

The obvious responses both fail. Waiting for a bigger sample is unaffordable. Publishing
the noisy number anyway means publishing a ranking of districts that is mostly noise:
next year's "worst district" will be somewhere else, for no reason other than sampling.

Small area estimation is the third response. It asks what else is known about a district
besides the few people who happened to be sampled there.

## Background

Write the direct estimate for area $i$ as $y_i$, and the quantity you actually want as
$\theta_i$. The survey gives you

$$y_i = \theta_i + e_i, \qquad e_i \sim N(0, D_i)$$

where $D_i$ is the **sampling variance**, known from the survey design rather than
estimated from the model. This is the crucial asymmetry: a small area has a large $D_i$,
and you know it is large.

The model half says the true values are not unrelated to each other. Areas with similar
characteristics tend to have similar outcomes:

$$\theta_i = x_i^\top \beta + u_i, \qquad u_i \sim N(0, \sigma_u^2)$$

Here $x_i$ holds auxiliary variables — administrative records, a census, satellite
data — known for *every* area, not just the sampled ones. The term $u_i$ admits that the
regression will not be perfect and leaves room for genuine area-specific variation.

Those two equations together are the Fay-Herriot model (Fay & Herriot, 1979), the
area-level foundation this package's Gaussian family implements. Rao & Molina (2015) is
the standard treatment of the field.

## The reasoning

The model has two independent sources of information about $\theta_i$: the survey, whose
precision is $1/D_i$, and the regression, whose precision is $1/\sigma_u^2$. Bayes' rule
combines them in proportion to their precision. For the Fay-Herriot model the posterior
mean works out to a weighted average:

$$\hat\theta_i = \gamma_i \, y_i + (1 - \gamma_i) \, x_i^\top \beta,
\qquad \gamma_i = \frac{\sigma_u^2}{\sigma_u^2 + D_i}$$

Everything worth understanding is in $\gamma_i$.

When $D_i$ is small — a well-sampled area — $\gamma_i$ approaches 1 and the estimate is
essentially the direct estimate. The model barely intervenes, because the survey already
knows the answer.

When $D_i$ is large — few respondents — $\gamma_i$ approaches 0 and the estimate is
pulled towards $x_i^\top \beta$, the value the auxiliary variables predict. The survey's
contribution is discounted in proportion to how noisy it is.

This is **borrowing strength**: the badly-sampled area borrows from the pattern
established across all the areas. Nothing is invented. The weight is not a tuning
parameter either — it falls out of the two variances, one known from the design and one
estimated from the data.

### Why shrinkage is not cheating

The direct estimate is unbiased and the shrunk estimate is not. That sounds like a step
backwards until you notice which quantity matters. Decisions are made worse by *error*,
and error has two parts:

$$\text{MSE} = \text{bias}^2 + \text{variance}$$

Shrinkage trades a small, controlled amount of bias for a large reduction in variance.
When $D_i$ is large the variance term dominates so heavily that accepting some bias
lowers the total. That is why the trade is worth making precisely where the direct
estimate is weakest, and why it barely happens where the direct estimate is strong.

Unbiasedness is a property of a procedure repeated infinitely often. You are publishing
one number for one district, once.

### Why hierarchical, and why Bayesian

*Hierarchical* is the structure: individual areas at one level, the parameters
$\beta$ and $\sigma_u^2$ they share at the level above. Those shared parameters are what
lets one area's data inform another's estimate.

*Bayesian* is how the uncertainty is handled. $\sigma_u^2$ is not known; it is estimated
from the same data. Classical approaches plug in an estimate and then have to correct the
resulting intervals, because pretending an estimated variance is known makes intervals
too narrow. Sampling from the joint posterior propagates that uncertainty automatically —
the credible interval for $\theta_i$ already accounts for not knowing $\sigma_u^2$.

The cost is that the posterior has no closed form for most families, so it is
approximated by MCMC. That is why every estimate in this package arrives with
convergence diagnostics attached, and why they are not optional.

## Consequences

**Your estimates will not match the direct estimates, and should not.** If they do, the
model contributed nothing. Compare the two and look for the pattern: the gap should be
widest for the areas with the largest sampling variance.

**Auxiliary variables carry the whole benefit.** If $x_i$ does not predict $\theta_i$,
then $\sigma_u^2$ is large, $\gamma_i$ stays near 1, and you get the direct estimates
back with extra steps. Choosing auxiliary variables that are genuinely related to the
outcome *and* available for every area is the substantive work; the fitting is
mechanical.

**A published estimate needs its uncertainty next to it.** Shrinkage narrows the interval
but does not abolish it. An area with a small sample still deserves a caveat, which is
why the estimation output carries the relative standard error alongside the mean.

**Known sampling variances have to be real.** $D_i$ is treated as known rather than
estimated. Supplying a rough guess in its place quietly changes how much each area
borrows, and nothing in the output will say so.

## Related

The procedure for turning a fitted model into a table of area estimates, including which
columns to read and how to check that shrinkage actually happened, is in
{doc}`../how-to/05-estimate-areas`.

## References

- Fay, R.E. & Herriot, R.A. (1979). Estimates of income for small places: an application
  of James-Stein procedures to census data. *Journal of the American Statistical
  Association*, 74(366), 269–277.
- Rao, J.N.K. & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
- Choir, A.S. et al. (2025). *hbsaems: Hierarchical Bayesian SAE Models*. R package.
