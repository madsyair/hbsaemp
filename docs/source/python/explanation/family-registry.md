# Why family metadata lives in one registry

Every fact about a family is written once, so the constructor and the running model always
agree.

## The question

`family="beta"` decides several things at once: which extra arguments are valid, which
links are allowed, the response domain, which columns the preprocessor derives, and the
name of the mean parameter in the posterior.

These facts are needed at two moments, when the model is built and while it runs, by code
in different modules. The obvious design stores each fact where it is used. This package
does not.

## Background

A family is a bundle of decisions that must stay consistent:

| Fact | Used by | Used when |
| --- | --- | --- |
| Which user arguments are valid | the constructor | at `create_model()` |
| Which links are supported | link validation | before sampling |
| Response domain and valid ranges | the validator | during `check_data()` |
| Which columns to derive | the preprocessor | during `check_data()` |
| Name of the mean parameter | prediction | after sampling |
| Backend family name | the model builder | at build time |

## The reasoning

### The risk is disagreement

If the constructor's list of valid arguments lives in one file and the preprocessor's list
of derived columns in another, updating one without the other gives a package that accepts
an argument and then ignores it.

Nothing raises. The user passes `deff="deff"`, the constructor accepts it, the preprocessor
never derives the precision column, and the model fits a different model from the one
requested.

With one registry, both read the same entry and cannot disagree.

### One source, two readers

`FamilySpec` is read by the factory, which needs to know what to accept before a model
exists, and by the running model, which needs the same facts afterwards.

The test: **adding a family means adding an entry**, plus the behaviour specific to that
family, not editing several files.

### Why cross-family arguments raise

The registry knows which arguments belong to `gaussian`, so it knows `trials` does not.
`create_model(..., family="gaussian", trials="n")` raises instead of dropping the argument.
Dropping it would give the caller a Gaussian model when they meant a Binomial one, with no
warning and plausible-looking estimates.

### Why the registry is read through accessors

The registry dict is private; `list_families()` and `get_family_spec()` are public. Its
internal shape can therefore change without breaking callers, and an unknown name gets an
error listing the valid families instead of a bare `KeyError`. Reading is supported,
mutating is not: a mutated registry would put the two readers out of step.

### Why a planned family gets its own message

The registry can mark a name as known but not yet implemented. Asking for it gives a message
saying so, not the error for a typo. A typo is fixed by correcting the spelling; a planned
family by waiting or choosing another.

## Consequences

**A family change is a registry change.** No fact about a family is stated twice.

**Errors name the alternatives.** The valid families and arguments are known at the moment
of failure, so the message says what would work.

**Validation runs before the backend loads.** The registry is plain Python data, so family,
link and argument checks finish in milliseconds.

**A new family is small but not free.** The entry is easy; the behaviour it promises, the
transformations and the formula shape, is the real work.

## Related

How to choose a family and supply its columns: {doc}`../how-to/02-specify-model`.
