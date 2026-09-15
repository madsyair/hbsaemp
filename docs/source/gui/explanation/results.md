# Why Results bundles comparison, refit, and estimation

Four things that sound separate all happen to the same object: the model
currently in use.

## The question

The dashboard's top-level navigation has only four tabs — Data, Data Exploration,
Modeling, Results — yet Results alone contains four sub-tabs: Convergence
Evaluation, Model Comparison, Update Model, and SAE Estimation. Update Model used
to be its own top-level tab, sitting fifth in the row. Why was it folded into
Results instead of left where it was, or split out further?

## Background

Each of these four things does something different: read diagnostics, rank saved
models, refit with different settings, compute small-area estimates. They also
depend on different inputs — Model Comparison reads `state.saved_models`, the
other three read `state.model`. Nothing about *what they do* obviously groups them
together.

## The reasoning

**They all act on the model currently in use, not on a stage of building one.**
Data, Data Exploration, and Modeling are each a step you pass through once per
analysis, in order — you upload data, you look at it, you specify and fit a model.
Convergence Evaluation, Model Comparison, Update Model, and SAE Estimation aren't
steps like that: they're things you might do zero, one, or several times, in any
order, to whichever model is currently active. Grouping them under one heading
reflects that they're a *set of operations on state.model*, not a sequence.

**A shared reset makes the grouping self-consistent.** `ResultsTab` watches
`state.model` once, in one place, and clears the R-hat/ESS table, plot pane, and
SAE table together the moment the model changes — including a change caused by
`Update Model` refitting it, or `Model Comparison`'s "Use Selected Model" swapping
it out. If these lived as separate top-level tabs, each would need its own watcher
on the same signal, and keeping four independent reset handlers in sync is exactly
the kind of duplication that quietly drifts.

**A fifth top-level tab competing with "Modeling" invites the wrong question.**
When Update Model sat beside Modeling in the top navigation, the natural question
was "do I refit here, or go back to Modeling?" — as if they were alternatives.
They aren't: Modeling builds a model from scratch, Update Model adjusts one that
already exists. Nesting the refit control inside Results, next to the diagnostics
it invalidates, makes that difference visible instead of asking the reader to
infer it from two same-looking tabs.

## Consequences

**Adding a new "thing to do with the active model" is a new sub-tab, not a new
top-level tab.** If a future feature reads or writes `state.model` after fitting —
sensitivity analysis, an export format, anything downstream of a fit — the
question to ask is whether it belongs in this same group, not whether it deserves
its own place in the main navigation.

**A genuinely new stage of the workflow — something that happens once, in
sequence, before or after the existing four — is the case for a new top-level
tab.** The dividing line is the one above: sequence versus repeatable
action-on-state.

## Related

For the click-by-click walkthrough of the sub-tabs described here, see
{doc}`../how-to/index` (dedicated guides for Update Model and Model Comparison
land with the remaining how-to pages).