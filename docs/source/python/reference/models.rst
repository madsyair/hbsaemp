Models
======

The three model interfaces, the objects they return, and the family registry.

.. currentmodule:: hbsaemp

Constructors
------------

A beginner shortcut such as :func:`hbm_gaussian` calls :func:`hbm_flex` (the
intermediate interface), which calls :func:`create_model` (the advanced
interface).

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

Each R-style alias, named after the R package ``hbsaems``, is the same
function as its Python name.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - R-style alias
     - Python name
   * - ``hbm``
     - :func:`create_model`
