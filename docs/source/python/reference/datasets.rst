Datasets
========

The bundled datasets, and the data-layer classes that validate and transform a
frame before a model sees it.

There are two kinds, both with one row per small area. The **synthetic**
datasets in :data:`AVAILABLE_DATASETS` have 30 areas and are generated from a
fixed seed (42): loading the same name twice gives the same numbers on any
machine. The **real** datasets in :data:`REAL_DATASETS` cover the 42 districts
and cities of Papua Island and are read from CSV files shipped with the package,
exactly as published — defects included, so that preparing them is part of the
case-study tutorial rather than hidden from it.

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

   Names of the synthetic datasets:
   ``["data_fhnorm", "data_betalogitnorm", "data_binlogitnorm", "data_lnln"]``.
   Every one of them loads ready to model.

.. data:: REAL_DATASETS
   :type: list[str]

   Names of the real datasets: ``["susenas2023_papua", "podes2021_papua"]``.
   :func:`load_dataset` accepts them too. They are kept out of
   :data:`AVAILABLE_DATASETS` because, as shipped, they need preparing before a
   model can use them. Any name in neither list raises :class:`ValueError`.

Two columns every synthetic dataset shares
------------------------------------------

``group`` and ``sre`` are both the area identifier, numbered 1 to 30, and they
hold **identical values** in all four synthetic datasets. Pass either one as ``group=``.
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

Real data: Papua Island
-----------------------

Two frames describing the same 42 districts and cities (*kabupaten/kota*) of
Papua Island, within the two provinces as bounded in 2021, Papua and Papua
Barat. They are the data of the case-study tutorial, which follows Prayoga,
Pusponegoro, Sukim and Budiarti (2024), *Small area estimation for morbidity
rate prediction*, Commun. Math. Biol. Neurosci. 2024:62,
https://doi.org/10.28919/cmbn/8845.

susenas2023_papua
~~~~~~~~~~~~~~~~~

Direct estimates of the 2023 morbidity rate: the percentage of the population
with a health complaint in the past month that disrupted their daily
activities. Source: Statistics Indonesia (BPS), National Socio-Economic Survey
(Susenas) March 2023, for the provinces of Papua and Papua Barat. Shape
``(42, 6)``.

.. list-table::
   :header-rows: 1
   :widths: 16 10 46 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``idkab``
     - int64
     - BPS district code, as published
     - [9101, 9437]
   * - ``nama_kab``
     - str
     - District or city name
     -
   * - ``nama_prov``
     - str
     - Province, 2021 boundaries
     - Papua, Papua Barat
   * - ``morbidity_rate``
     - float64
     - Direct estimate of the morbidity rate, in percent
     - [1.01, 13.77]
   * - ``se``
     - float64
     - Standard error of ``morbidity_rate``, in percentage points
     - [0.27, 2.41]
   * - ``rse``
     - float64
     - Relative standard error of ``morbidity_rate``, in percent
     - [9.09, 60.63]

.. warning::

   Two records are shipped exactly as published, and neither is model-ready:

   - **Deiyai** (9436) has no ``morbidity_rate`` and no ``se``, only an
     ``rse`` of 60.63.
   - **Kota Jayapura** is coded 9437 here. Its official code, used by
     ``podes2021_papua``, is 9471, so a merge on ``idkab`` drops it.

podes2021_papua
~~~~~~~~~~~~~~~

Auxiliary variables from the 2021 Village Potential Data Collection (Podes
2021, BPS), aggregated from the village records — *desa* and *kelurahan* — of
each district. Shape ``(42, 18)``.

.. list-table::
   :header-rows: 1
   :widths: 16 10 46 28

   * - Column
     - dtype
     - Meaning
     - Range as shipped
   * - ``idkab``
     - int64
     - BPS district code
     - [9101, 9471]
   * - ``nama_kab``
     - str
     - District or city name
     -
   * - ``kode_prov_2021``, ``nama_prov_2021``
     - int64, str
     - Province before the 2022 split
     - 91 Papua Barat, 94 Papua
   * - ``kode_prov_2022``, ``nama_prov_2022``
     - int64, str
     - Province after the 2022 split
     - six provinces, codes [91, 97]
   * - ``n_kec``
     - int64
     - Number of subdistricts (*kecamatan*)
     - [5, 51]
   * - ``n_desa``
     - int64
     - Number of villages, *desa* and *kelurahan* together
     - [38, 545]
   * - ``X1`` … ``X10``
     - float64
     - Auxiliary variables, defined below
     - see below

Each auxiliary variable is computed from Podes 2021 village items over **all**
villages of the district, *desa* and *kelurahan* alike. The item codes are those
of the Podes 2021 village questionnaire. Every value in the file was recomputed
from the village records with the definitions below and matches to rounding.

.. list-table::
   :header-rows: 1
   :widths: 8 44 30 18

   * - Name
     - Meaning
     - Computed from
     - Range as shipped
   * - ``X1``
     - Percentage of villages where most families defecate in latrines
     - share with ``r505a`` in {1, 2, 3}
     - [0, 100]
   * - ``X2``
     - Percentage of villages where most families drink from a decent water
       source
     - share with ``r507a`` in {1, 2, 3, 4, 5, 6, 7, 9}
     - [14.56, 100]
   * - ``X3``
     - Percentage of villages where most families bathe and wash with water
       from a decent source
     - share with ``r507b`` in {1, 2, 3, 4, 5, 6, 7}
     - [98.42, 100]
   * - ``X4``
     - Public elementary or equivalent schools per village
     - mean of ``r701dk2 + r701ek2``
     - [0.0769, 1.44]
   * - ``X5``
     - Public junior high or equivalent schools per village
     - mean of ``r701fk2 + r701gk2``
     - [0.0232, 0.436]
   * - ``X6``
     - Public senior high or equivalent schools per village
     - mean of ``r701hk2 + r701ik2 + r701jk2``
     - [0.00917, 0.539]
   * - ``X7``
     - Hospitals (general and maternity) and community health centres
       (*puskesmas*) per village
     - mean of ``r704ak2 + r704bk2 + r704ck2 + r704dk2``
     - [0.0232, 0.590]
   * - ``X8``
     - Integrated health posts (*posyandu*) with service activities once a
       month or more, per village
     - mean of ``r705b``
     - [0, 5.49]
   * - ``X9``
     - Doctors and midwives per village
     - mean of ``r706a1 + r706a2 + r706c``
     - [0.0431, 5.49]
   * - ``X10``
     - Other health workers per village
     - mean of ``r706d``
     - [0.0388, 5.92]

.. note::

   Four village records hold the value 99 in a health-worker item: ``r706a2``
   in Bawei (Biak Numfor), and ``r706d`` in Wouyebutu (Paniai), Bis Agats
   (Asmat) and Kasonaweja (Mamberamo Raya). Whether 99 is a count or a special
   code has not been verified. The records are aggregated as recorded, so they
   feed ``X9`` for Biak Numfor and ``X10`` for Paniai, Asmat and Mamberamo Raya.
