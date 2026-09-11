Estimation
==========

Per-area estimates from a fitted model, and refitting an existing model
specification against new data, a changed formula or priors, or a new sampler
configuration.

.. currentmodule:: hbsaemp

Area estimates
--------------

.. autosummary::
   :toctree: generated/

   estimate_areas
   AreaEstimatesResult

Refitting
---------

.. autosummary::
   :toctree: generated/

   update_model
   update_formula

R aliases
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - R-style alias
     - Python name
   * - ``hbsae``
     - :func:`estimate_areas`
   * - ``update_hbm``
     - :func:`update_model`
