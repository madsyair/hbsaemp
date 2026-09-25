# Why there are three ways to build the same model

Each interface asks for one decision fewer than the one above it.

## The question

Three functions build the same model. `hbm_gaussian()` takes a response and a list of
auxiliary variables. `hbm_flex()` takes the same plus a family. `create_model()` takes a
formula. Which one is the real API, and what are the others for?

## Background

The R package offers a single entry point in the style of `caret::train()`: one function,
with a family argument selecting the model.

That design assumes the caller writes model formulas. Someone who knows they have a
proportion, three auxiliary variables and a district column may not know the formula
language, and should not need it.

## The reasoning

The three interfaces form one chain, each link removing a decision:

```
hbm_gaussian()  →  hbm_flex()  →  create_model()  →  the model object
```

| Interface | You supply | It decides for you |
| --- | --- | --- |
| Beginner: `hbm_beta`, `hbm_gaussian`, `hbm_binomial` | response, auxiliary list, the family's own columns | the family and the formula |
| Intermediate: `hbm_flex` | response, auxiliary list, family | the formula |
| Advanced: `create_model` / `hbm` | the formula | nothing |

`create_model` starts with `formula`, `family`, `data`. `hbm_flex` replaces the formula with
`response` and `auxiliary`. `hbm_gaussian` also drops `family` and exposes `sampling_var`,
the extra column a Gaussian small area model needs.

### Where a family's own arguments live

Only the beginner interface names a family's columns: `hbm_beta` takes `n` and `deff`, and
`hbm_binomial` takes `trials`. `hbm_flex` passes them through two generic arguments:

```python
hbm_binomial("y", ["x1"], df, trials="n")                        # beginner
hbm_flex("y", ["x1"], df, family="binomial", addition_var="n")   # the same model
hbm_beta("y", ["x1"], df, n="n", deff="deff")                    # beginner
hbm_flex("y", ["x1"], df, family="beta",
         aux_args={"n": "n", "deff": "deff"})                    # the same model
```

`addition_var` is the column the likelihood needs beside the response, and `aux_args` holds
the rest. A new family declares its columns in the registry and reaches `hbm_flex` through
these two arguments, so `hbm_flex` never changes. At the beginner interface the names stay
explicit, so a typo is an immediate `TypeError`.

### Why each interface delegates

Each interface calls the one after it. The beginner interface fixes the family and calls
`hbm_flex`, which builds the formula and calls `create_model`, where validation, group
injection and construction happen.

Three separate implementations would be three places for behaviour to drift. A validation
rule added to `create_model` but not to `hbm_beta` would make the same model behave
differently depending on the function used. Delegation rules that out.

### Why the signatures are narrow

No interface accepts a catch-all `**kwargs`. Every argument is named, and each shortcut
exposes only its own family's arguments.

`hbm_gaussian(..., trials="n")` is therefore a `TypeError` raised by Python at the call,
naming the argument. A permissive signature would pass the mistake several layers down, or
ignore it and build a different model from the one requested.

### Why the advanced interface still exists

The beginner and intermediate interfaces can add only one random-effect term: the random
intercept `(1|area)`, through `area_var=`. Any other term, such as a random slope `(x1|group)`, needs a formula, and a
formula needs `create_model`. Transformations and interactions are computed as columns at
every interface.

## Consequences

**Start with the simplest interface that can express your model.** It has the fewest
arguments to get wrong. Move to the next one when you need something it cannot express.

**A translated R script uses the advanced interface.** `hbm()` is `create_model()`, so an
`hbsaems` call carries over unchanged.

**`model.formula` is the formula that is fitted.** Whatever the interface, it shows the
resolved formula, including the group term added for you.

**Family-specific columns are checked against the family** at every interface, so a stray
argument fails the same way everywhere.

## Related

How to specify a model with each interface: {doc}`../how-to/02-specify-model`.
