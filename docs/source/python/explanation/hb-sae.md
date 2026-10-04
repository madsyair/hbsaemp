# What hierarchical Bayes small area estimation is

Why a direct estimate fails a small area, and how a hierarchical Bayes model borrows strength
from the other areas.

## The question

A survey is designed to give reliable figures for large domains, such as a province. Broken
down by district, each district keeps only a few respondents. An area is *small* when its
sample is too small for a precise direct estimate, whatever its geographic size
(Pfeffermann, 2013; Rao & Molina, 2015).

The direct estimate of such an area is still design-unbiased, but its variance grows as the
area's sample shrinks. Statistics Indonesia (BPS) grades an estimate by its relative standard
error (RSE), the standard error as a percentage of the estimate (Badan Pusat Statistik, 2025):

| RSE | Reading |
| --- | --- |
| 25% or less | precise |
| 25% to 50% | usable, with caution |
| above 50% | imprecise |

A bigger sample is too expensive. Small area estimation asks what else is known about a
district besides its few sampled respondents.

## Background

### Direct, synthetic and composite estimators

Rao & Molina (2015) distinguish three kinds of estimator:

- A **direct** estimator uses only the area's own sample. It is unbiased under the design,
  and noisy when the sample is small.
- A **synthetic** estimator applies a relationship fitted on all areas, such as a regression
  on auxiliary variables. It is stable, but biased wherever an area departs from the common
  relationship, and that bias does not shrink as the sample grows.
- A **composite** estimator is a weighted average of the two.

A model-based estimator is a composite whose weight follows from a statistical model.

### The Fay-Herriot model

The area-level model of Fay & Herriot (1979) is the standard example. Its **sampling model**
links the direct estimate $y_i$ of area $i$ to the true value $\theta_i$:

$$y_i = \theta_i + e_i, \qquad e_i \sim N(0, \psi_i),$$

where the sampling variance $\psi_i$ is known from the survey design. Its **linking model**
relates the true values to auxiliary variables $x_i$ available for every area, such as
census or administrative counts:

$$\theta_i = x_i^\top \beta + v_i, \qquad v_i \sim N(0, \sigma_v^2).$$

The area effect $v_i$ absorbs what the auxiliary variables do not explain. This is the model
of the Gaussian family. `sampling_var=` names the column holding $\psi_i$, which is `D` in the
bundled `data_fhnorm`.

### Three levels

Written as a hierarchy, the model has three levels:

1. the sampling model, $y_i \mid \theta_i \sim N(\theta_i, \psi_i)$;
2. the linking model, $\theta_i \mid \beta, \sigma_v^2 \sim N(x_i^\top \beta, \sigma_v^2)$;
3. a prior on the shared parameters, $p(\beta, \sigma_v^2)$.

The third level makes the model hierarchical Bayes. Unless `priors=` says otherwise, hbsaemp
uses Bambi's weakly informative priors there: normal priors on the coefficients and a
half-normal prior on $\sigma_v$, both scaled to the data (Capretto et al., 2022).

## The reasoning

### The weight follows from two variances

For known $\beta$ and $\sigma_v^2$, the best predictor of $\theta_i$ is

$$\tilde\theta_i = \gamma_i \, y_i + (1 - \gamma_i) \, x_i^\top \beta,
\qquad \gamma_i = \frac{\sigma_v^2}{\sigma_v^2 + \psi_i}.$$

It is a composite estimator whose weight $\gamma_i$ is not set by hand:

- **Small $\psi_i$** (a well-sampled area): $\gamma_i$ is close to 1, and the estimate stays
  close to the direct estimate.
- **Large $\psi_i$** (few respondents): $\gamma_i$ is close to 0, and the estimate moves
  towards $x_i^\top \beta$, the value the auxiliary variables predict.

This is **borrowing strength**: a poorly sampled area borrows from the pattern fitted across
all areas, and the less its own sample says, the more it borrows.

### Why shrinkage reduces error

The direct estimate is unbiased; the shrunk estimate is not. What matters for a decision is
the total error, $\text{MSE} = \text{bias}^2 + \text{variance}$. Shrinkage accepts a small
bias for a large reduction in variance. The trade is largest where $\psi_i$ is largest, which
is where the direct estimate is weakest.

### Why hierarchical Bayes rather than EBLUP

In practice $\beta$ and $\sigma_v^2$ are unknown. The frequentist EBLUP estimates them first,
plugs them into $\tilde\theta_i$, and corrects its MSE for that step afterwards. Its intervals
often fall short of their nominal coverage (Rao & Molina, 2015).

Hierarchical Bayes treats $\beta$ and $\sigma_v^2$ as random too. The estimate of an area is
its posterior mean, the shrinkage predictor above averaged over the posterior of the shared
parameters:

$$\hat\theta_i^{\text{HB}} = E(\theta_i \mid y)
= E_{\beta, \sigma_v^2}\big[E(\theta_i \mid y, \beta, \sigma_v^2) \mid y\big].$$

Its posterior variance adds the uncertainty about $\beta$ and $\sigma_v^2$ to the uncertainty
left after shrinkage, so no separate correction is needed.

Two more properties matter here. The sampling model can follow the indicator, so a proportion
or a count gets a likelihood of its own instead of a normal approximation. And the posterior
has no closed form, so it is sampled by MCMC; every estimate therefore depends on the checks
described in {doc}`workflow-order`.

### When the indicator is a proportion

A rate such as morbidity lies in $(0, 1)$, while a normal sampling model allows any value. The
Beta-logistic model of Liu et al. (2014) uses a Beta sampling model in mean and precision
form:

$$y_i \mid \mu_i \sim \text{Beta}\big(\mu_i \phi_i, \, (1 - \mu_i) \phi_i\big),
\qquad \operatorname{Var}(y_i \mid \mu_i) = \frac{\mu_i (1 - \mu_i)}{1 + \phi_i},$$

and links the area proportion to the auxiliary variables through the logit:

$$\operatorname{logit}(\mu_i) = x_i^\top \beta + v_i, \qquad v_i \sim N(0, \sigma_v^2).$$

Here $\mu_i$ is the true area proportion, and the precision
$\phi_i = n_i / \text{deff}_i - 1$ comes from the area's sample size and design effect. Bambi
calls the mean `mu` and the precision `kappa`, and `n=` with `deff=` fixes `kappa` at
$\phi_i$. This is the model of the case-study tutorial, after Prayoga et al. (2024).

Shrinkage works the same way on the logit scale. The sampling variance of
$\operatorname{logit}(y_i)$ is approximately $1 / \big((1 + \phi_i) \, \mu_i (1 - \mu_i)\big)$,
close to the squared RSE when $\mu_i$ is small. The weight an area keeps on its own direct
estimate therefore depends on its RSE, which involves its rate as well as $\phi_i$: a high
precision does not protect a very small rate from shrinkage.

The Binomial family covers a count of successes $y_i$ out of $n_i$ trials, with
$\operatorname{logit}(p_i) = x_i^\top \beta + v_i$. Its sampling variance follows from $n_i$
and $p_i$.

### Why the sampling precision is fixed, not estimated

Each area contributes one direct estimate. If the sampling variance were unknown, one
observation per area could not separate it from the area effect: the data identify only the
total variance $\sigma_v^2 + \psi_i$. The survey design breaks the tie. It gives $\psi_i$, or
$n_i$ and $\text{deff}_i$, from the sample itself, so the model fixes them and estimates
$\sigma_v^2$.

Without them, Bambi estimates one scale or one precision shared by all areas, and every area
looks equally precise. The weight $\gamma_i$ then no longer follows the area's own sample,
which is the property small area estimation rests on. That is why `sampling_var=`, and `n=`
with `deff=`, fix a parameter of the likelihood; {doc}`family-registry` describes how.

### Why the estimate summarises $\theta_i$, not $y_i$

The target is the true value $\theta_i$. The posterior predictive distribution describes a
new direct estimate $y_i$, which adds the sampling error $e_i$ back in. Summarising it would
restore the noise the model was built to remove. The area estimates therefore summarise the
posterior of the mean parameter: `mu` for Gaussian and Beta, `p` for Binomial.

Each area gets its posterior mean as the estimate, its posterior standard deviation, the RSE
computed from the two, and a highest-density interval (HDI), the shortest interval holding
the stated posterior probability. The `mse` column holds the posterior variance, the measure
of uncertainty in hierarchical Bayes. For a symmetric posterior the HDI equals the
equal-tailed interval; for the skewed posterior of a small proportion it is shorter and sits
closer to the mode.

## Consequences

**Each estimate moves a share of the way, set by $\gamma_i$.** An estimate moves the fraction
$1 - \gamma_i$ of the distance from $y_i$ to $x_i^\top \beta$. The distance itself depends on
how far the direct estimate lies from the regression, so the largest moves need not be in the
areas with the largest $\psi_i$. If every estimate equals its direct estimate, the model added
nothing.

**The auxiliary variables carry the benefit.** If $x_i$ does not predict $\theta_i$,
$\sigma_v^2$ is large, $\gamma_i$ stays near 1, and the direct estimates come back. Choosing
auxiliary variables related to the outcome and available for every area is the substantive
work.

**The sampling variances must be real.** $\psi_i$ and $\phi_i$ are treated as known. A rough
guess changes how much each area borrows, and the output does not show it.

**An area with no sample still gets an estimate.** It is the limit $\gamma_i = 0$: the
synthetic value $x_i^\top \beta$, with an area effect taken at each posterior draw from a
fitted area. Its interval is wider than a sampled area's, because no data narrow its own
effect.

**Every published estimate needs its uncertainty.** Shrinkage narrows the interval but does
not remove it. Report the RSE and the interval with each estimate, and keep an area above 25%
in the table with its precision class.

**The default priors are a choice.** Other priors, such as the flat prior on $\beta$ and the
inverse-gamma prior on $\sigma_v^2$ common in the small area literature, can give slightly
different estimates. The prior checks in {doc}`workflow-order` show whether the choice matters
for your data.

## Related

How to produce the area estimates and check the shrinkage: {doc}`../how-to/09-estimate-areas`.

## References

- Badan Pusat Statistik. (2025). *Tata kelola dan kerangka kerja small area estimation*.
  BPS.
- Capretto, T. et al. (2022). Bambi: A simple interface for fitting Bayesian linear models
  in Python. *Journal of Statistical Software*, 103(15).
- Fay, R. E. & Herriot, R. A. (1979). Estimates of income for small places: an application
  of James-Stein procedures to census data. *Journal of the American Statistical
  Association*, 74(366), 269–277.
- Liu, B., Lahiri, P., & Kalton, G. (2014). Hierarchical Bayes modeling of survey-weighted
  small area proportions. *Survey Methodology*, 40(1), 1–13.
- Pfeffermann, D. (2013). New important developments in small area estimation. *Statistical
  Science*, 28(1), 40–68.
- Prayoga, I. Y. A., Pusponegoro, N. H., Sukim, & Budiarti, W. (2024). Small area estimation
  for morbidity rate prediction. *Communications in Mathematical Biology and Neuroscience*,
  2024:62.
- Rao, J. N. K. & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
