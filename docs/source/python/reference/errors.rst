Errors and warnings
===================

Every exception the package raises on purpose, arranged by the hierarchy they
form. Catching :class:`HBSAEError` catches all of them except
:class:`ConvergenceWarning`, which is a warning rather than an error.

.. admonition:: Not written yet
   :class: stub

   The table of which call raises which error, and what to do about each one,
   is part of the documentation structure. Its content lands in v0.2.

.. currentmodule:: hbsaemp

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

.. STRUCTURE -- do not delete; fill in and remove these comments when writing.
.. ## <error class>   one section per class
.. ##                 an attribute table, and which call raises it
.. RULE  this page states facts. The recovery procedure belongs in a how-to page.
