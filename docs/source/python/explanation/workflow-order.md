# Why the Bayesian workflow has this order

Each stage checks something the next stage relies on, so the order carries statistical
meaning.

## The question

The How-to guides are numbered. It is tempting to treat the numbering as a convenience: fit
the model, read the estimates, and check the diagnostics only if something looks odd.

That shortcut turns a hierarchical Bayes model into a black box, which returns numbers
without the checks that make them usable (Gelman et al., 2020). MCMC does not report failure.
A sampler that explored only part of the posterior still returns draws, and their means,
intervals and rankings look like trustworthy ones. Only the checks tell them apart.

## Background

Box's loop, as Blei (2014) summarises it, cycles through building a model, computing the
posterior, criticising the model against the data and revising it. The Bayesian workflow of
Gelman et al. (2020) covers the same cycle of model building, inference, and model checking
and improvement, and adds two steps to it: validating the computation and comparing models.
A failed check sends the analysis back to an earlier stage.

hbsaemp turns each stage into one function call, in this order:

| Stage | Function (R alias) | In the Bayesian workflow |
| --- | --- | --- |
| 1 · Load the data | `load_dataset`, or pandas | input to the workflow |
| 2 · Specify the model | `hbm_<family>`, `hbm_flex`, `create_model` (`hbm`) | building a model |
| 3 · Check the priors | `check_prior` (`hbpc`) | prior predictive check |
| 4 · Fit the model | `fit()` | fitting the model |
| 5 · Check convergence | `check_convergence` (`hbcc`) | validating the computation |
| 6 · Update the model | `update_model` (`update_hbm`) | addressing computational problems |
| 7 · Check the fit and compare models | `compare_models` (`hbmc`) | posterior predictive check, cross-validation, comparing models |
| 8 · Check prior sensitivity | `compare_models` with `run_prior_sensitivity=True` | influence of the prior |
| 9 · Estimate the areas | `estimate_areas` (`hbsae`) | using the model |

Stages 3 and 7 send you back to stage 2 when they fail, and so does a prior-data conflict
found in stage 8. Stage 5 sends you to stage 6 and then back to stage 5.

## The reasoning

### The model is specified before anything is sampled

Building a model does not sample. `check_data()` runs the validation and preprocessing of
`fit()` on its own, so a missing column or a proportion outside $(0, 1)$ shows before any MCMC
is paid for. An error costs least at the start.

### Priors are checked before fitting

A prior predictive check simulates data from the priors alone and asks whether they are
plausible for the domain (Gelman et al., 2020). If the model considers a morbidity rate of
300% plausible, you should know before fitting.

The check must come first. Once the posterior is known, any prior chosen afterwards is partly
chosen to reproduce it. The check is only honest while the result is unknown.

### Convergence is checked before any result is read

R-hat compares chains that started in different places, which is why a run needs several
chains. The effective sample size (ESS) counts how many independent draws the chains are
worth. Divergent transitions mark regions the sampler could not follow. All three describe
the sampling, and none is visible in the estimates. `check_convergence` warns when R-hat
exceeds 1.01 or the bulk or tail ESS falls below 100 per chain (Vehtari et al., 2021), and on
any divergence.

An estimate read before this check may describe where the sampler got stuck rather than the
posterior. Reading the estimates first also biases you: you now know which verdict you want
from the diagnostics.

### Computational trouble is information

Gelman et al. (2020) note that a computational problem often points to a problem with the
model. The remedies therefore escalate: first the sampler settings, through `update_model`
with a higher `target_accept`, then more tuning, then more draws; when those fail, the
specification. A refit gives a new posterior, so it returns to the convergence check. The
earlier verdict applied to the old one.

### The fit is checked before models are compared

A posterior predictive check asks whether the model can reproduce the data it was fitted to,
by setting the observed data against data replicated from the posterior, usually by eye
(Gabry et al., 2019). A model that cannot is misspecified, and ranking it against others
ranks the wrong candidates. The fix is a different specification, so the workflow returns to
stage 2.

Comparison then scores how well each candidate predicts data it has not seen, by PSIS-LOO
cross-validation (Vehtari et al., 2017). It needs the log-likelihood of every observation,
which `fit()` computes after sampling for this reason. In an area-level model every area
contributes one observation and has its own random effect, so leaving that observation out
removes the only direct information about the effect. Pareto *k* values above the threshold
are therefore common, and when most observations are flagged the ranking is not reliable.

Comparison must come after convergence, because comparing unreliable draws gives a
reliable-looking ranking of nothing. It must come before estimation, because choosing the
model whose estimates you like fits the analyst's expectations instead of the data.

### Prior sensitivity is checked on the chosen model

The prior check in stage 3 asks whether the priors are plausible. The sensitivity check asks
whether they matter. Power-scaling raises the prior, and separately the likelihood, to a power
close to 1 and measures how far each posterior moves, reusing the existing draws through
importance sampling instead of refitting (Kallioinen et al., 2024). It needs a posterior, so
it comes after fitting, and it is run on the model that will be reported.

A parameter fixed by the survey design is expected to be flagged as a strong prior with a
weak likelihood: the tight prior on its intercept is how it is fixed.

### Estimation comes last

An area estimate relies on every earlier stage: plausible priors, a converged sampler, an
adequate and preferred model, and a known influence of the prior. The estimate is only as
good as that chain of checks.

## Consequences

**A slow stage is still a required one.** Convergence checking produces no publishable
output, which is why it gets skipped. The package makes each check an explicit call, so
skipping one is a decision.

**Warnings are signals.** A `ConvergenceWarning` is the check doing its job. It is a
`UserWarning`, so a script keeps running; silencing it removes the only sign that a posterior
is unusable.

**Two loops, two remedies.** Poor convergence returns you to the sampler settings. Implausible
priors, a poor fit and a prior-data conflict return you to the specification.

**Going back is normal.** Each stage must be passed before the next is trusted, however many
times you pass through it.

**The case study runs the same workflow.** The tutorial is arranged by the six phases of
CRISP-DM (Chapman et al., 2000). Its data understanding and data preparation phases deliver
stage 1, and its modeling and evaluation phases run stages 2 to 9.

## Related

The stages, one page each: {doc}`../how-to/index`.

## References

- Blei, D. M. (2014). Build, compute, critique, repeat: Data analysis with latent variable
  models. *Annual Review of Statistics and Its Application*, 1, 203–232.
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. CRISP-DM
  Consortium.
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in
  Bayesian workflow. *Journal of the Royal Statistical Society Series A*, 182(2), 389–402.
- Gelman, A. et al. (2020). Bayesian workflow. arXiv:2011.01808.
- Kallioinen, N., Paananen, T., Bürkner, P.-C., & Vehtari, A. (2024). Detecting and
  diagnosing prior and likelihood sensitivity with power-scaling. *Statistics and
  Computing*, 34, 57.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: An improved $\widehat{R}$ for assessing
  convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
