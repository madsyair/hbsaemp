Diagnostics
===========

Checks run before and after sampling: prior predictive checks, convergence
diagnostics, and model comparison.

.. currentmodule:: hbsaemp

Prior checks
------------

.. autosummary::
   :toctree: generated/

   check_prior
   PriorCheckResult

Convergence
-----------

.. autosummary::
   :toctree: generated/

   check_convergence
   ConvergenceResult

Model comparison
----------------

.. autosummary::
   :toctree: generated/

   compare_models
   ComparisonResult

R aliases
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - R-style alias
     - Python name
   * - ``hbpc``
     - :func:`check_prior`
   * - ``hbcc``
     - :func:`check_convergence`
   * - ``hbmc``
     - :func:`compare_models`
