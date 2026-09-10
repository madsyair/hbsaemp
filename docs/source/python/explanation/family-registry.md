# Why family metadata lives in one registry

Every fact about a family is written once, so the constructor and the running model can
never disagree about it.

## The question

Choosing `family="beta"` settles a surprising number of things at once. Which extra
arguments are legal. Which link functions are allowed. What the response domain is.
Which columns the preprocessor must derive. What the mean parameter is called in the
posterior.

Those facts are needed at two very different moments — when the model is built, and while
it runs — and by code in different modules. The obvious implementation puts each fact
where it is used. This package deliberately does not.

## Background

A family is not a single switch. It is a small bundle of related decisions that have to
stay consistent with each other:

| Fact | Used by | Used when |
| --- | --- | --- |
| Which user arguments are legal | the constructor | at `create_model()` |
| Which links are supported | link validation | before sampling |
| Response domain and admissible ranges | the validator | during `check_data()` |
| Which columns to derive | the preprocessor | during `check_data()` |
| Name of the mean parameter | prediction | after sampling |
| Backend family name | the model builder | at build time |

Spread these across the modules that consume them and each one looks perfectly
reasonable in isolation.

## The reasoning

### The failure mode is disagreement, not duplication

Duplication is not the real problem; drift is. If the constructor's list of legal
arguments lives in one file and the model's idea of which attributes exist lives in
another, adding a column to one and not the other produces a package that accepts an
argument and then ignores it.

That failure is quiet. Nothing raises. The user passes `deff="deff"`, the constructor
accepts it because its list was updated, the preprocessor never derives the precision
column because its list was not, and the model fits — a different model from the one that
was asked for, reporting no problem at all.

A single registry makes the two halves read from the same source, so they cannot hold
different opinions.

### One source, two very different readers

`FamilySpec` is read by the factory, which needs to know what to accept before any model
exists, and by the running model, which needs the same facts about itself afterwards.
Both consult the same entry.

The practical test of whether this is working: **adding a family should be adding an
entry, not editing several files.** When metadata is centralised, a new family is one
registry entry plus whatever genuinely family-specific *behaviour* it needs. When it is
scattered, it is an archaeology exercise across modules, and the thing you forget is
always the one with no test.

### Why cross-family arguments are refused rather than ignored

Because the registry knows exactly which arguments belong to `gaussian`, it can also say
with certainty that `trials` does not. So `create_model(..., family="gaussian",
trials="n")` raises instead of silently dropping the argument.

Ignoring it would be friendlier in the moment and much worse in an hour. The caller
believed they specified a Binomial model. Accepting the argument and discarding it means
they get a Gaussian model, no warning, and a set of estimates that look entirely
plausible.

### Why the registry is reached through accessors

The registry dict itself is not part of the public surface. `list_families()` and
`get_family_spec()` are.

That boundary is what allows the internal shape to change — a field added, a
representation altered — without breaking anyone. It also means an unknown name gets a
proper error naming the families that do exist, rather than a bare `KeyError`. Reading is
supported; reaching in and mutating is not, because a mutated registry would desynchronise
exactly the two readers this design exists to keep in step.

### Why a planned family can be refused informatively

The registry can also record that a name is *known but not implemented*. Asking for a
family planned for a later version gives a message saying so, rather than the same error
as a typo. Those are different mistakes and deserve different answers — one is corrected
by fixing a spelling, the other by waiting or choosing something else.

## Consequences

**Changing a docstring changes the documentation; changing a family changes the
registry.** Nothing about a family is stated twice, so there is nothing to keep in sync.

**The error you get names the alternatives.** Because the list of valid families and
valid arguments is available at the moment of failure, messages can say what would have
worked instead of only what did not.

**Validation happens before the modelling backend is involved.** The registry is plain
Python data, so the family, link and argument checks all run before anything heavyweight
is imported. Mistakes surface in milliseconds.

**A new family is a small change, but not a free one.** Adding an entry is the easy part;
supplying the behaviour that entry promises — the transformations, the formula shape —
is the real work. The registry keeps that work honest by making it obvious what has been
promised.

## Related

The procedure for choosing a family and supplying its required columns, including what
each family accepts, is in {doc}`../how-to/01-specify-model`.
