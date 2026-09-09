Datasets
========

The bundled example datasets, and the data-layer classes that validate and
transform a frame before a model sees it.

Every dataset has one row per small area, 30 areas, and is generated from a
fixed seed (42). Loading the same name twice gives the same numbers on any
machine, which is what makes the tutorials reproducible.

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

   Names accepted by :func:`load_dataset`:
   ``["data_fhnorm", "data_betalogitnorm", "data_binlogitnorm", "data_lnln"]``.
   Any other name raises :class:`ValueError`.

Two columns every dataset shares
--------------------------------

``group`` and ``sre`` are both the area identifier, numbered 1 to 30, and they
hold **identical values** in all four datasets. Pass either one as ``group=``.
The second name exists to mirror the R ``hbsaems`` package, where ``sre``
labels the spatial random effect.

.. warning::

   Columns whose name ends in ``_true``, plus ``theta``, ``u`` and
   ``mu_orig_true``, are the **values used to generate the data**. Real survey
   data has no such columns. They are shipped so you can check an estimate
   against the truth — never use them as auxiliary variables, or the model will
   look excellent and mean nothing.

data_fhnorm
-----------

Gaussian Fay-Herriot model with known sampling variances. Shape ``(30, 9)``.

.. math::

   y_i = \theta_i + e_i, \quad e_i \sim N(0, D_i)

   \theta_i = x_i^\top \beta + u_i, \quad u_i \sim N(0, \sigma_u^2)

Use with ``family="gaussian"`` and ``sampling_var="D"``.

.. list-table::
   :header-rows: 1
   :widths: 14 12 46 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``y``
     - float64
     - Direct estimate for the area
     - [3.16, 6.24]
   * - ``D``
     - float64
     - Known sampling variance of ``y``; must be positive
     - [0.109, 0.483]
   * - ``x1``, ``x2``, ``x3``
     - float64
     - Auxiliary variables
     - [-1.95, 2.14]
   * - ``theta_true``
     - float64
     - True area mean — generated only
     - [3.33, 6.32]
   * - ``u``
     - float64
     - True area random effect — generated only
     - [-0.864, 0.801]
   * - ``group``, ``sre``
     - int64
     - Area identifier
     - [1, 30]

data_betalogitnorm
------------------

Beta logit-normal model for a proportion. Shape ``(30, 9)``.

.. math::

   y_i \sim \text{Beta}(\mu_i \phi_i, (1 - \mu_i)\phi_i)

   \text{logit}(\mu_i) = x_i^\top \beta + u_i, \quad u_i \sim N(0, \sigma_u^2)

   \phi_i = n_i / \text{deff}_i - 1

Use with ``family="beta"``, ``n="n"`` and ``deff="deff"``. The precision
:math:`\phi_i` is derived from the survey design rather than estimated, which
is why ``n`` and ``deff`` must be supplied together or not at all.

.. list-table::
   :header-rows: 1
   :widths: 14 12 46 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``y``
     - float64
     - Direct proportion, strictly inside (0, 1)
     - [0.139, 0.617]
   * - ``theta``
     - float64
     - True mean :math:`\mu_i` — generated only
     - [0.145, 0.625]
   * - ``x1``, ``x2``, ``x3``
     - float64
     - Auxiliary variables
     - [-1.95, 2.14]
   * - ``n``
     - float64
     - Effective sample size; must be positive
     - [53, 198]
   * - ``deff``
     - float64
     - Design effect; needs ``n / deff > 1``
     - [1.23, 2.42]
   * - ``group``, ``sre``
     - int64
     - Area identifier
     - [1, 30]

data_binlogitnorm
-----------------

Binomial logit-normal model for a count out of a known number of trials.
Shape ``(30, 14)``.

.. math::

   y_i \sim \text{Binomial}(n_i, p_i)

   \text{logit}(p_i) = x_i^\top \beta + u_i, \quad u_i \sim N(0, \sigma_u^2)

Use with ``family="binomial"`` and ``trials="n"``.

.. list-table::
   :header-rows: 1
   :widths: 14 12 46 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``n``
     - int64
     - Number of trials
     - [32, 119]
   * - ``y``
     - int64
     - Successes; non-negative integers, never above ``n``
     - [5, 45]
   * - ``p``
     - float64
     - Direct proportion, ``y / n``
     - [0.124, 0.721]
   * - ``x1``, ``x2``, ``x3``
     - float64
     - Auxiliary variables
     - [-1.95, 2.14]
   * - ``u_true``, ``eta_true``, ``p_true``
     - float64
     - True effect, linear predictor and probability — generated only
     - [-0.864, 0.801] / [-1.95, 0.670] / [0.125, 0.662]
   * - ``psi_i``
     - float64
     - Sampling variance of the logit-scale direct estimate
     - [0.0345, 0.235]
   * - ``y_obs``, ``p_obs``
     - float64
     - ``y`` and ``p`` again, as floats
     - [5, 45] / [0.124, 0.721]
   * - ``group``, ``sre``
     - int64
     - Area identifier
     - [1, 30]

.. note::

   ``y_obs`` and ``p_obs`` duplicate ``y`` and ``p``. Model the integer ``y``
   with ``trials="n"``; the float copies exist to mirror the R package's column
   names.

data_lnln
---------

Lognormal-lognormal model. Shape ``(30, 13)``.

.. warning::

   **This dataset loads, but no family in this version can model it.**
   ``load_dataset("data_lnln")`` returns a frame like any other, yet
   ``create_model(..., family="lognormal")`` raises
   :class:`ModelRegistryError` — *"family='lognormal' is planned for V2 and is
   not available in V1"* — and :class:`DataValidator` rejects the family
   outright. It is documented here only because the name is visible in
   :data:`AVAILABLE_DATASETS`, so finding it and getting no explanation would
   be worse.

.. math::

   \log(y_i) \sim N(\eta_i, \psi_i)

   \eta_i = x_i^\top \beta + u_i, \quad u_i \sim N(0, \sigma_u^2)

.. list-table::
   :header-rows: 1
   :widths: 16 12 44 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``y_log_obs``
     - float64
     - Log-scale direct estimate; the intended response
     - [0.855, 2.91]
   * - ``lambda_dir``
     - float64
     - Same values as ``y_log_obs``
     - [0.855, 2.91]
   * - ``y_obs``
     - float64
     - Direct estimate back on the original scale
     - [2.35, 18.4]
   * - ``psi_i``
     - float64
     - Known sampling variance on the log scale
     - [0.00277, 0.0128]
   * - ``n``
     - float64
     - Area sample size
     - [32, 148]
   * - ``x1``, ``x2``, ``x3``
     - float64
     - Auxiliary variables
     - [-1.95, 2.14]
   * - ``u_true``, ``theta_true``, ``mu_orig_true``
     - float64
     - True effect, log-scale mean and original-scale mean — generated only
     - [-0.691, 0.641] / [0.688, 2.99] / [2.16, 21.4]
   * - ``group``, ``sre``
     - int64
     - Area identifier
     - [1, 30]
