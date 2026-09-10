# Why there are three ways to build the same model

The same model can be written three ways, and each tier hides one more decision than the
tier below it.

## The question

Three functions produce the same fitted model. `hbm_gaussian()` takes a response and a
list of auxiliary variables. `hbm_flex()` takes the same plus a family. `create_model()`
takes a formula string. Offering three doors into one room looks like indecision, and
raises a fair question: which one is the real API, and what are the others for?

## Background

The R package this one is ported from offers a single entry point, in the style of
`caret::train()` — one function, one call, a family argument selecting the model. That
design is deliberate and works well.

It also assumes the caller is comfortable writing model formulas. A formula is a compact
language, and compactness is only an advantage once you already know the language. For
someone who knows they have a proportion, three auxiliary variables and a district
column, the formula is an obstacle placed in front of an otherwise clear intention.

## The reasoning

The three tiers are not alternatives. They are one chain, each link removing a decision:

```
hbm_gaussian()  →  hbm_flex()  →  create_model()  →  the model object
```

| Tier | You supply | It decides for you |
| --- | --- | --- |
| 3 — `hbm_beta`, `hbm_gaussian`, `hbm_binomial` | response, auxiliary list, the family's own columns | the family, and the formula |
| 2 — `hbm_flex` | response, auxiliary list, family | the formula |
| 1 — `create_model` / `hbm` | the formula itself | nothing |

Compare the signatures and the pattern is visible without reading any prose.
`create_model` opens with `formula`, `family`, `data`. `hbm_flex` replaces the formula
with `response` and `auxiliary`. `hbm_gaussian` drops `family` too, and in its place
exposes `sampling_var` — the one extra column a Gaussian small area model actually needs.

### Why delegation rather than three implementations

Each tier calls the one below it. Tier 3 assembles nothing itself; it fixes the family
and hands over to tier 2, which builds the formula and hands over to tier 1, where the
validation, the group injection and the construction live.

The alternative — three functions that each build a model their own way — would mean
three places for behaviour to drift apart. A validation rule added to `create_model` and
forgotten in `hbm_beta` produces a package where the same model behaves differently
depending on which door you came through. Keeping the chain intact makes that impossible
rather than merely discouraged.

### Why the narrow signatures are the point

The tiers do not accept a catch-all `**kwargs`. Every argument is named explicitly, and
each family's shortcut exposes only that family's extras.

The consequence is that `hbm_gaussian(..., trials="n")` is a `TypeError` from Python
itself, raised at the call, naming the argument. Had the signatures been permissive, the
same mistake would have travelled several layers deep before failing as something
obscure — or worse, been silently ignored, leaving you with a model that quietly is not
the one you asked for.

A narrow signature is the cheapest possible validation: it costs nothing at runtime and
the error message writes itself.

### Why tier 1 still exists

Tier 3 cannot express everything. Interactions, an intercept you want suppressed, a
random slope rather than a random intercept — these need the formula language, and the
formula language needs tier 1. Removing it would trade a small gain in tidiness for a
ceiling on what the package can express.

## Consequences

**Pick the highest tier that can say what you mean.** Higher tiers have fewer ways to go
wrong, because there are fewer things to get wrong. Drop a tier when you hit something it
cannot express, not on principle.

**A translated R script lands on tier 1.** `hbm()` is the same function as
`create_model()`, so an `hbsaems` call carries over unchanged. Tier 2 and 3 are additions
this port makes available, not replacements.

**The formula you get is the formula that is used.** Whichever tier you enter through,
`model.formula` shows the resolved string, including a group term added on your behalf.
There is no hidden second formula.

**Family-specific columns are checked against the family.** They are declared once per
family and enforced everywhere, which is why a stray argument fails the same way whether
you came in at tier 1 or tier 3.

## Related

The procedure for specifying a model at any tier, including what each family requires and
what happens when a formula term is rejected, is in
{doc}`../how-to/01-specify-model`.
