# Why the Bayesian workflow has this order

The stages are a sequence, and the order carries statistical meaning.

## The question

The how-to guides are numbered, and the numbering looks like a convention — a tidy way
to arrange pages rather than a claim about statistics. It is tempting to treat it that
way: fit the model, look at the estimates, and go back to the diagnostics only if
something seems odd.

That shortcut is the single most common way to produce a confident number that means
nothing. The order is not editorial. Each stage answers a question that makes the next
stage's answer interpretable, and running them out of order produces output that looks
identical to correct output.

## Background

The workflow this package follows is the one that has become standard practice for
applied Bayesian analysis: specify, check the priors, fit, diagnose the sampler, compare
candidate models, and only then read the estimates. The stages map onto the how-to
guides one for one.

What makes the order binding is that MCMC does not report failure. A sampler that has
explored only part of the posterior returns draws, and those draws produce means,
intervals and rankings that are formatted exactly like trustworthy ones. Nothing in the
output distinguishes them. The diagnostics are the only thing that does.

## The reasoning

### Priors are checked before the data is used

A prior predictive check asks what the model believes before it has seen any outcome
data: draw parameters from the prior, generate data from them, and look at what comes
out. If the model considers a poverty rate of 300% plausible, that is worth knowing
before fitting rather than after.

This has to happen *before* fitting for a reason that is easy to miss. Once you have seen
the posterior, you cannot un-see it, and any prior you choose afterwards is chosen partly
to produce the answer you already saw. The check is only meaningful while you are still
ignorant of the result.

### Convergence is checked before any estimate is read

This is the stage most often skipped, and the one with the sharpest consequence.

R-hat compares chains that started in different places. If they explored the same
distribution, their summaries agree and R-hat is near 1. Effective sample size asks how
many *independent* draws the correlated chain is worth. Both are properties of the
sampling, not of the model — and both are invisible in the estimates themselves.

Reading an estimate before checking convergence means you have no idea whether the number
describes the posterior or describes where the sampler happened to get stuck. If you look
at the estimates first and the diagnostics second, you have also created a subtler
problem: you now know which answer you want the diagnostics to permit.

### Comparison comes before estimation, not after

Model comparison scores how well each candidate predicts data it has not seen. Doing this
after choosing estimates inverts the logic — you would be selecting the model whose
estimates you liked, which is a way of fitting the analyst's expectations rather than the
data.

Comparison also has to come after convergence, and for the same reason as everything
else: comparing two sets of untrustworthy draws yields a trustworthy-looking ranking of
nothing at all.

### Estimation is last because it consumes everything before it

By the time you read an area estimate you are relying on every earlier stage at once: the
priors were reasonable, the sampler explored the posterior, and this specification was
the best of those considered. The estimate is a summary of the posterior, and the
posterior is only as good as the process that produced it.

### Refitting re-enters the sequence, it does not continue it

Changing the sampler settings or the data produces a new posterior, and every diagnostic
verdict from before applies to the old one. A refit therefore re-enters at the
diagnostics stage rather than resuming where the previous fit stopped. Carrying an old
"converged" verdict across a refit is the same mistake as skipping the check, arrived at
more politely.

## Consequences

**A slow stage is not an optional stage.** Convergence checking on a large model takes
time and produces no publishable output, which is exactly why it gets dropped. The
package makes it explicit rather than automatic, so skipping it is a decision you make
rather than one that happens quietly.

**Warnings are load-bearing.** A `ConvergenceWarning` is not noise to be filtered; it is
the stage doing its job. Silencing it removes the only signal that separates a usable
posterior from an unusable one.

**Escalate in a fixed order.** When diagnostics fail, raise `target_accept` first, then
`tune`, then `draws`. That order follows the same logic as the workflow: fix how the
sampler explores before buying more of what it already produced.

**Going backwards is normal.** The sequence is not a one-way pipeline. Poor diagnostics
send you back to the sampler settings, and a comparison that favours a different
specification sends you back to the beginning. What matters is that each stage is passed
before the next is trusted, not that you pass through them only once.

## Related

The procedure for the fitting stage itself, including how to choose sampler settings and
what to do when they are not enough, is in {doc}`../how-to/02-fit-model`.

## References

- Rao, J.N.K. & Molina, I. (2015). *Small Area Estimation* (2nd ed.). Wiley.
- Capretto, T. et al. (2022). Bambi: A simple interface for fitting Bayesian linear
  models in Python. *Journal of Statistical Software*, 103(15).
