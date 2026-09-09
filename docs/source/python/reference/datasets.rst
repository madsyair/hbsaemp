Datasets
========

The bundled example datasets, and the data-layer classes that validate and
transform a frame before a model sees it.

.. admonition:: Not written yet
   :class: stub

   The per-dataset column tables are part of the documentation structure.
   Their content lands in v0.2.

.. currentmodule:: hbsaemp

Loading
-------

.. autosummary::
   :toctree: generated/

   load_dataset

Data layer
----------

.. autosummary::
   :toctree: generated/

   DataValidator
   DataPreprocessor

Module constants
----------------

.. data:: AVAILABLE_DATASETS
   :type: list[str]

   Names accepted by :func:`load_dataset`. Every dataset has 30 rows and is
   generated from a fixed seed, so results are reproducible across machines.

.. STRUCTURE -- do not delete; fill in and remove these comments when writing.
.. ## <dataset name>   one section per dataset
.. ##                  a column table: name, dtype, meaning, admissible range
.. ##                  which family the dataset is intended for
.. RULE  this page states facts. No procedure, no reasoning.
