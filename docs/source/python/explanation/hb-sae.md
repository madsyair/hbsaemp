# What hierarchical Bayes small area estimation is

Why a direct estimate is unreliable for a small area, and how borrowing strength helps.

## The question

A survey is designed for reliable national figures. Broken down by district, each district
keeps only a few respondents. The district's direct estimate is still unbiased, but its
variance is too large to support a decision.

A bigger sample is too expensive. Publishing the noisy number means publishing a ranking
that is mostly noise: next year's "worst district" will be another one, by chance alone.

Small area estimation asks what else is known about a district besides its few sampled
respondents.

## Background

Write the direct estimate for area $i$ as $y_i$ and the true value as $\theta_i$. The survey
gives

$$y_i = \theta_i + e_i, \qquad e_i \sim N(0, D_i)$$

where $D_i$ is the **sampling variance**, known from the survey design. A small area has a
large $D_i$, and you know it.

The model links the true values through auxiliary variables $x_i$, such as administrative
records or a census, available for every area:

$$\theta_i = x_i^\top \beta + u_i, \qquad u_i \sim N(0, \sigma_u^2)$$

The random effect $u_i$ allows for area-specific variation the regression does not explain.

Together these two equations are the Fay-Herriot model (Fay & Herriot, 1979), which the
Gaussian family of this package implements. Rao & Molina (2015) is the standard reference.

## The reasoning

The model has two sources of information about $\theta_i$: the survey, with precision
$1/D_i$, and the regression, with precision $1/\sigma_u^2$. Bayes' rule weights them by
precision. For the Fay-Herriot model the posterior mean is

$$\hat\theta_i = \gamma_i \, y_i + (1 - \gamma_i) \, x_i^\top \beta,
\qquad \gamma_i = \frac{\sigma_u^2}{\sigma_u^2 + D_i}$$

- **Small $D_i$** (a well-sampled area): $\gamma_i$ is close to 1, and the estimate stays
  close to the direct estimate.
- **Large $D_i$** (few respondents): $\gamma_i$ is close to 0, and the estimate moves towards
  $x_i^\top \beta$, the value the auxiliary variables predict.

This is **borrowing strength**: a poorly sampled area borrows from the pattern across all
areas. The weight is not tuned by hand. It follows from two variances, one known from the
design and one estimated from the data.

### Why shrinkage reduces error

The direct estimate is unbiased; the shrunk estimate is not. What matters for a decision is
the total error:

$$\text{MSE} = \text{bias}^2 + \text{variance}$$

Shrinkage accepts a small bias for a large reduction in variance. When $D_i$ is large, the
variance dominates, so the total error falls. The trade happens mostly where the direct
estimate is weakest.

### Why hierarchical, and why Bayesian

*Hierarchical* is the structure: areas at one level, and the parameters they share,
$\beta$ and $\sigma_u^2$, at the level above. The shared parameters let one area's data
inform another area's estimate.

*Bayesian* is how uncertainty is handled. $\sigma_u^2$ is estimated from the same data.
Plugging in an estimate and treating it as known makes intervals too narrow. Sampling from
the joint posterior carries that uncertainty into the credible interval of every
$\theta_i$.

The posterior has no closed form for most families, so it is approximated by MCMC. That is
why every estimate comes with convergence diagnostics.

### Why the estimate summarises $\theta_i$, not $y_i$

The target is the true value $\theta_i$. The posterior predictive distribution describes a
new direct estimate $y_i$, which adds the sampling error $e_i$ back in. Summarising it would
restore the noise the model was built to remove, so the area estimates summarise the
posterior of $\theta_i$.

## Consequences

**The estimates differ from the direct estimates.** If they match, the model contributed
nothing. The gap should be widest where the sampling variance is largest.

**The auxiliary variables carry the benefit.** If $x_i$ does not predict $\theta_i$,
$\sigma_u^2$ is large, $\gamma_i$ stays near 1, and the direct estimates come back. Choosing
auxiliary variables related to the outcome and available for every area is the substantive
work.

**Every published estimate needs its uncertainty.** Shrinkage narrows the interval but does
not remove it. The estimation output therefore reports the relative standard error with the
mean.

**The sampling variances must be real.** $D_i$ is treated as known. A rough guess changes
how much each area borrows, and the output does not show it.

## Related

How to produce the area estimates and check the shrinkage: {doc}`../how-to/09-estimate-areas`.

## References

- Fay, R.E. & Herriot, R.A. (1979). Estimates of income for small places: an application
  of James-Stein procedures to census data. *Journal of the American Statistical
  Association*, 74(366), 269–277.
- Rao, J.N.K. & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
- Choir, A.S. et al. (2025). *hbsaems: Hierarchical Bayesian SAE Models*. R package.
