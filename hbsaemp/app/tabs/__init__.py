"""Tab controllers for the hbsaemp Panel dashboard.

Each tab is a ``param.Parameterized`` object holding a reference to the
shared :class:`~hbsaemp.app._state.AppState`, and exposes a ``.panel()``
method returning its Panel layout.
"""

from __future__ import annotations

from hbsaemp.app.tabs.data_tab import DataTab
from hbsaemp.app.tabs.explore_tab import ExploreTab
from hbsaemp.app.tabs.model_tab import ModelTab
from hbsaemp.app.tabs.results_tab import ResultsTab

__all__: list[str] = ["DataTab", "ExploreTab", "ModelTab", "ResultsTab"]