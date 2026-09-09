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
   from :class:`UserWarning`, not from :class:`HBSAEError`, because it reports
   a result worth distrusting rather than an operation that failed. Sampling
   finished; the answer may just be unreliable. Catch it with
   :class:`warnings.catch_warnings`, not with ``except``.

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
     - ``create_model()``, ``check_data()``, ``fit()``
   * - :class:`FormulaError`
     - The formula string cannot be parsed, or a term is not a bare column name
     - ``create_model()``, ``hbm_flex()``
   * - :class:`PriorSpecError`
     - A prior specification is malformed
     - ``Prior(...)``, ``priors=`` in ``create_model()``
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
     - R-hat above 1.01, or bulk/tail ESS below 400
     - ``check_convergence()``

Two of these are worth reading twice. :class:`FormulaError` is raised rather
than silently dropping a term: an in-formula transform such as ``np.log(x1)``
or ``x1:x2`` is rejected, because a dropped term would hide a predictor from
validation and resurface later as an obscure backend error. Pre-compute it as a
DataFrame column instead. And :class:`ModelNotFittedError` is deliberately
*not* wrapped by :class:`EstimationError`, so "called in the wrong order" stays
distinguishable from "this model and data cannot be estimated".

.. note::

   ``summary()`` is the exception to the rule above: calling it before ``fit()``
   returns a short "not fitted" description instead of raising, so it is always
   safe to print a model to see what it will do.

Not raised in this version
--------------------------

Three names in the public API never appear as the *cause* of a failure here.
They are listed because they are importable and part of the advertised
surface, not because you should expect to see them.

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
diagnostics, safe to log). Some add one more field naming the thing that was
wrong.

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
