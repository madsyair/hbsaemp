# Why the Bayesian workflow has this order

Each stage makes the next one interpretable, so the order carries statistical meaning.

## The question

The How-to guides are numbered. It is tempting to treat the numbering as a convenience: fit
the model, read the estimates, and check the diagnostics only if something looks odd.

That shortcut produces confident numbers that mean nothing. Output from stages run out of
order looks exactly like correct output.

## Background

This package follows the workflow used in applied Bayesian analysis:

1. **Load the data**, one row per area.
2. **Specify the model**: family, auxiliary variables, area identifier and priors.
3. **Check the priors.** If the data they imply is implausible, return to 2.
4. **Fit the model.**
5. **Check convergence.** If the chains have not converged, refit (6) and check again.
6. **Update the model**, as the remedy for 5.
7. **Check the fit and compare models.** If the model does not reproduce the data, return
   to 2.
8. **Check prior sensitivity.**
9. **Estimate the areas.**

MCMC does not report failure. A sampler that explored only part of the posterior still
returns draws, and their means, intervals and rankings look like trustworthy ones. Only the
checks tell them apart.

## The reasoning

### Priors are checked before fitting

A prior predictive check simulates data from the priors alone. If the model considers a
morbidity rate of 300% plausible, you should know before fitting.

The check must come first. Once you have seen the posterior, any prior you choose is partly
chosen to reproduce it. The check is only honest while the result is unknown.

### Convergence is checked before any result is read

R-hat compares chains that started in different places; effective sample size counts how
many independent draws the chains are worth. Both describe the sampling, and neither is
visible in the estimates.

An estimate read before this check may describe where the sampler got stuck rather than the
posterior. Reading the estimates first also biases you: you now know which verdict you want
from the diagnostics.

### A refit returns to the convergence check

New sampler settings or new data give a new posterior. The earlier verdict applied to the
old one, so a refit goes back to stage 5.

### The fit is checked before models are compared

A posterior predictive check asks whether the model can reproduce the data it was fitted
to. A model that cannot is misspecified, and ranking it against others ranks the wrong
candidates. The fix is a different specification, so the workflow returns to stage 2.

Comparison scores how well each candidate predicts data it has not seen. It must come after
convergence, because comparing unreliable draws gives a reliable-looking ranking of
nothing, and before estimation, because choosing the model whose estimates you like fits the
analyst's expectations instead of the data.

### Prior sensitivity is checked on the chosen model

The prior check in stage 3 asks whether the priors are plausible. The sensitivity check
asks whether they matter: how far the posterior moves when the prior is scaled. It needs a
posterior, so it comes after fitting, and it is run on the model that will be reported.

### Estimation comes last

An area estimate relies on every earlier stage: plausible priors, a converged sampler, an
adequate and preferred model, and a known influence of the prior. The estimate is only as
good as that chain of checks.

## Consequences

**A slow stage is still a required one.** Convergence checking produces no publishable
output, which is why it gets skipped. The package makes each check an explicit call, so
skipping one is a decision.

**Warnings are signals.** A `ConvergenceWarning` is the check doing its job. Silencing it
removes the only sign that a posterior is unusable.

**Escalate in a fixed order.** When convergence fails, raise `target_accept`, then `tune`,
then `draws`: fix how the sampler explores before buying more of the same draws.

**Going back is normal.** Poor priors and an inadequate model return you to stage 2; poor
convergence returns you to the sampler settings. Each stage must be passed before the next
is trusted, however many times you pass through it.

## Related

- The stages, one page each: {doc}`../how-to/index`.
- Choosing sampler settings: {doc}`../how-to/04-fit-model`.

## References

- Rao, J.N.K. & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
- Capretto, T. et al. (2022). Bambi: A simple interface for fitting Bayesian linear
  models in Python. *Journal of Statistical Software*, 103(15).
