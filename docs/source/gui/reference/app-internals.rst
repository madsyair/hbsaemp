App internals
=============

Auto-generated from docstrings in ``hbsaemp.app``. This page is for
contributors extending the dashboard — for how to *use* the app, see the
:doc:`../how-to/index` guides instead.

Launching
---------

.. currentmodule:: hbsaemp.app

.. autosummary::
   :toctree: generated/

   launch_app
   App
   AppConfig

Shared state
------------

The reactive state passed to every tab, and the snapshot type used to
freeze a model for later comparison.

.. currentmodule:: hbsaemp.app._app

.. autosummary::
   :toctree: generated/

   AppState
   SavedModel

Tabs
----

One class per tab, in the order they appear in the dashboard.

.. currentmodule:: hbsaemp.app.tabs

.. autosummary::
   :toctree: generated/

   DataTab
   ExploreTab
   ModelTab
   ResultsTab
   UpdateModelTab

Model tab widgets
-----------------

Supporting widget used inside :class:`~hbsaemp.app.tabs.model_tab.ModelTab`.

.. currentmodule:: hbsaemp.app.tabs.model_tab

.. autosummary::
   :toctree: generated/

   PredictorCheckboxes