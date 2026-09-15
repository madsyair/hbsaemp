# Why the app doesn't gate on convergence

The numbers are shown as-is. Deciding what they mean is left to you.

## The question

Convergence diagnostics look like a pass/fail test: R-hat close to 1, effective
sample size above some threshold, no divergences. It would be easy for the app to
read those numbers itself and refuse to let you save, compare, or trust a model
that didn't clear them — an earlier version of this app did exactly that. Why was
that removed?

## Background

At one point, saving a model recorded whether `check_convergence()` had raised a
`ConvergenceWarning`, and the Model Comparison table only let you *select* a model
that had converged when saved — attempting to compare an unconverged one returned
an error telling you to refit first. The intent was protective: stop a bad fit from
quietly feeding into a comparison or a final estimate.

That gating is gone. `SavedModel` now holds only the model itself, `check_convergence()`'s
own warning is caught only so it doesn't print to the server's console, and Model
Comparison will happily compare models regardless of their diagnostics.

## The reasoning

**The threshold is a convention, not a law.** R-hat ≤ 1.01 and ESS ≥ 100 × the number
of chains are commonly used defaults, not universal cutoffs — the same numbers that
are too strict for one study can be too lenient for another, depending on how the
estimates will be used downstream. Hardcoding that threshold into a gate means the
app is making a statistical judgment call on the analyst's behalf, using a rule of
thumb it has no way to calibrate to the actual study.

**A gate teaches the wrong lesson.** Once a check exists that blocks you when it
fails, passing it starts to feel like a guarantee — "the app let me save it, so it
must be fine." R-hat and ESS are properties of the *sampling*, not proof the model is
correct or the estimates are usable; treating them as a checkpoint to clear rather
than a diagnostic to read encourages exactly the shortcut the workflow is meant to
prevent.

**The person reading the numbers has context the app doesn't.** Whether a borderline
R-hat of 1.015 is acceptable depends on things no widget can know: how the estimates
will be reported, how much time is left to refit, whether the borderline parameter
even matters for the areas being estimated. Refusing to save or compare removes that
judgment call from the one person actually positioned to make it.

## Consequences

**Nothing stops you from comparing or exporting a fit with real convergence
problems.** The Convergence Evaluation tab is not a formality — it's the only place
the app tells you anything about diagnostics at all, and unlike before, it does so
without also acting on your behalf.

**Reading Convergence Evaluation is now something you have to remember to do**,
rather than something the app enforces. There is no badge, checkmark, or blocked
button reminding you. If you add a new feature that consumes a saved or fitted
model, it should follow the same pattern — show what's known, don't decide for the
user — rather than reintroducing a gate.

## Related

For where the R-hat/ESS numbers come from and how to read them, see
{doc}`../how-to/index` (a dedicated guide for the Results tab lands with the
remaining how-to pages).