Errors and warnings
===================

Every exception the package raises on purpose, arranged by the hierarchy they
form.

.. currentmodule:: hbsaemp

Catching them
-------------

:class:`HBSAEError` is the base of every error below, so one ``except`` clause
covers all of them::

    try:
        model = create_model("y ~ x1", family="gaussian", data=df)
        model.fit()
    except HBSAEError as err:
        print(err.message)
        print(err.context)

.. important::

   :class:`ConvergenceWarning` is **not** covered by that clause. It inherits
   from :class:`UserWarning`, not from :class:`HBSAEError`: sampling finished,
   but the result may be unreliable. Catch it with
   :class:`warnings.catch_warnings`.

Which call raises what
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Exception
     - Raised when
     - Typically from
   * - :class:`DataValidationError`
     - A column is missing, non-numeric, infinite, or outside the domain the
       family allows; or dropping missing rows would empty the frame
     - ``create_model()``, ``check_data()``, ``fit()``,
       ``predict(new_data=...)``, ``estimate_areas(new_data=...)``,
       ``update_model(new_data=...)``
   * - :class:`FormulaError`
     - The formula string cannot be parsed, or a term is not a bare column name
     - ``create_model()``, ``hbm_flex()``, ``update_model(formula=...)``
   * - :class:`PriorSpecError`
     - A prior specification is malformed
     - ``Prior(...)``, ``priors=`` in ``create_model()`` and ``update_model()``
   * - :class:`ModelRegistryError`
     - ``family=`` names something that is not registered, including a family
       planned for a later version
     - ``create_model()``, ``get_family_spec()``
   * - :class:`ModelNotFittedError`
     - A result is requested before sampling has happened
     - ``predict()``, ``estimate_areas()``, ``check_convergence()``,
       ``compare_models()``, ``update_model()``
   * - :class:`EstimationError`
     - Per-area estimation fails for a reason that is not one of the above
     - ``estimate_areas()``
   * - :class:`ConvergenceWarning`
     - R-hat above 1.01, bulk/tail ESS below 100 × chains, divergent
       transitions, maximum tree depth hits, or E-BFMI below 0.3
     - ``check_convergence()``

An in-formula transform such as ``np.log(x1)`` or ``x1:x2`` raises
:class:`FormulaError` instead of being dropped; compute it as a DataFrame column.
:class:`ModelNotFittedError` is not wrapped in :class:`EstimationError`, so a
call in the wrong order stays distinguishable from a model that cannot be
estimated.

.. note::

   ``summary()`` does not raise before ``fit()``: it returns a short
   "not fitted" description of the model.

Not raised in this version
--------------------------

These three names are importable but never raised directly.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Status
   * - :class:`HBSAEError`
     - Base class. Catch it; nothing raises it directly.
   * - :class:`ValidationError`
     - Groups the four input-validation errors. Catch it to handle any bad
       input in one place; nothing raises it directly.
   * - :class:`SpatialMatrixError`
     - Reserved for spatial random effects, which this version does not
       implement. No code path reaches it.

Attributes
----------

Every error carries ``.message`` (the text) and ``.context`` (a dict for
diagnostics, safe to log). Some add one more attribute:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Exception
     - Extra attribute
     - Holds
   * - :class:`DataValidationError`
     - ``.column``
     - Offending column name, when known
   * - :class:`FormulaError`
     - ``.formula``
     - The formula string that failed to parse
   * - :class:`PriorSpecError`
     - ``.param``
     - Parameter the prior was attached to
   * - :class:`SpatialMatrixError`
     - ``.matrix_type``
     - Which matrix was rejected
   * - :class:`ModelRegistryError`
     - ``.family``, ``.registered``
     - The unknown name, and the list of names that would have worked

Base class
----------

.. autosummary::
   :toctree: generated/

   HBSAEError

Validation errors
-----------------

Raised before sampling starts, while the formula, the priors and the data are
being checked.

.. autosummary::
   :toctree: generated/

   ValidationError
   DataValidationError
   FormulaError
   PriorSpecError
   SpatialMatrixError

Model and estimation errors
---------------------------

.. autosummary::
   :toctree: generated/

   ModelRegistryError
   ModelNotFittedError
   EstimationError

Warnings
--------

.. autosummary::
   :toctree: generated/

   ConvergenceWarning
