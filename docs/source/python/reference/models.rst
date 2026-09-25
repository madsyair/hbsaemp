Models
======

Model constructors for the three interface tiers, the objects they return, and
the family registry that drives them.

.. currentmodule:: hbsaemp

Constructors
------------

The three tiers cascade: a tier 1 shortcut (the beginner interface, such as
:func:`hbm_gaussian`) calls :func:`hbm_flex` (tier 2, the intermediate
interface), which calls :func:`create_model` (tier 3, the advanced interface).
Pick the tier that matches how much you want to spell out.

.. autosummary::
   :toctree: generated/

   create_model
   hbm_flex
   hbm_beta
   hbm_gaussian
   hbm_binomial

Model objects
-------------

.. autosummary::
   :toctree: generated/

   BaseModel
   ModelResult

Configuration and priors
------------------------

.. autosummary::
   :toctree: generated/

   ModelConfig
   Prior

Family registry
---------------

.. autosummary::
   :toctree: generated/

   FamilySpec
   list_families
   get_family_spec

Module constants
----------------

Described here rather than generated, because a bare ``dict`` or ``list``
carries no docstring of its own and autodoc would fall back to the built-in
type's documentation.

.. data:: DEFAULT_CONFIG
   :type: ModelConfig

   The :class:`ModelConfig` used when ``create_model()`` is called without a
   ``config=`` argument.

.. data:: MODEL_REGISTRY
   :type: dict[str, type[BaseModel]]

   Maps a family name to the :class:`BaseModel` subclass that
   :func:`create_model` instantiates for it.

R aliases
---------

The package mirrors the naming of the R package ``hbsaems``. Each alias is the
same object under a second name, so it takes the same arguments and returns the
same result.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - R-style alias
     - Python name
   * - ``hbm``
     - :func:`create_model`
