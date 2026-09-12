"""Tab 5 — Update Model: refit an already-fitted model.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
import param

from hbsaemp import (
    DataValidationError,
    EstimationError,
    FormulaError,
    HBSAEError,
    ModelNotFittedError,
    ModelRegistryError,
    PriorSpecError,
    update_model,
)
from hbsaemp._logging import get_logger
from hbsaemp.data.datasets import AVAILABLE_DATASETS, load_dataset

if TYPE_CHECKING:
    from hbsaemp.app._app import AppState
    from hbsaemp.models._base import BaseModel

logger = get_logger(__name__)

__all__: list[str] = ["UpdateModelTab"]


def _error_box(title: str, body: str = "") -> str:
    inner = f"<b>{title}</b><br>{body}" if body else title
    return (
        f'<div style="background:#d62728;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{inner}</div>'
    )


def _success_box(body: str) -> str:
    return (
        f'<div style="background:#2ca02c;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _info_box(body: str) -> str:
    return (
        f'<div style="background:#0072B2;color:white;padding:10px 16px;'
        f'border-radius:8px;margin-top:8px">{body}</div>'
    )


def _pending_box(message: str) -> str:
    return (
        f'<div style="background:#fff3cd;border-left:4px solid #d62728;'
        f'color:#7a1f1f;padding:10px 16px;border-radius:6px;'
        f'font-family:monospace;font-size:0.95em">{message}</div>'
    )


def _describe_error(exc: Exception) -> tuple[str, str]:
    """Map a backend exception to a (title, body) pair for the UI.

    Mirrors :func:`hbsaemp.app.tabs.model_tab._describe_error` — kept as a
    small local copy rather than a shared cross-tab helper module (see
    ``app/`` package layout).
    """
    if isinstance(exc, ModelNotFittedError):
        return "Model has not been fitted", "Please fit the model in the Modeling tab first."
    if isinstance(exc, FormulaError):
        return "Invalid formula", str(exc)
    if isinstance(exc, ModelRegistryError):
        return "Family not available", str(exc)
    if isinstance(exc, DataValidationError):
        return "Replacement data does not meet the family's requirements", str(exc)
    if isinstance(exc, PriorSpecError):
        return "Invalid prior specification", str(exc)
    if isinstance(exc, EstimationError):
        return "Estimation failed", str(exc)
    if isinstance(exc, ImportError):
        return "Missing dependency", str(exc)
    if isinstance(exc, HBSAEError):
        return "Backend error", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return "Invalid configuration", str(exc)
    return "Unexpected error", str(exc)


class UpdateModelTab(param.Parameterized):
    """Refit the currently-fitted model via :func:`hbsaemp.update_model`.

    Args:
        state: Shared :class:`~hbsaemp.app._app.AppState` instance. Reads
            ``state.model`` (written by
            :class:`~hbsaemp.app.tabs.model_tab.ModelTab`); after a
            successful refit, writes the same (mutated-in-place) model
            back to ``state.model`` and, if replacement data was used,
            ``state.data`` too.
    """

    state: AppState = param.Parameter()

    def __init__(self, state: AppState, **params: Any) -> None:
        super().__init__(state=state, **params)

        self._current_info = pn.pane.HTML(
            _pending_box("No fitted model yet — fit one in the Modeling tab first.")
        )
        self._draws_cb, self._draws_in = self._override_row(
            "draws", pn.widgets.IntInput(value=1000, start=1, max_width=140)
        )
        self._tune_cb, self._tune_in = self._override_row(
            "tune", pn.widgets.IntInput(value=1000, start=0, max_width=140)
        )
        self._chains_cb, self._chains_in = self._override_row(
            "chains", pn.widgets.IntInput(value=4, start=1, max_width=140)
        )
        self._cores_cb, self._cores_in = self._override_row(
            "cores", pn.widgets.IntInput(value=1, start=1, max_width=140)
        )
        self._target_accept_cb, self._target_accept_in = self._override_row(
            "target_accept",
            pn.widgets.FloatInput(value=0.9, start=0.01, end=0.99, step=0.01, max_width=160),
        )
        self._seed_cb, self._seed_in = self._override_row(
            "random_seed", pn.widgets.IntInput(value=42, start=0, max_width=140)
        )

        # Replacement data — optional.
        self._new_data_cb = pn.widgets.Checkbox(
            name="Refit on different data (formula/family/link/group stay the same)",
            value=False,
        )
        self._new_data_file = pn.widgets.FileInput(
            accept=".csv", name="Upload replacement CSV", disabled=True,
        )
        self._new_data_dataset_sel = pn.widgets.Select(
            name="...or pick a built-in dataset",
            options=[None, *AVAILABLE_DATASETS], value=None, disabled=True,
        )
        self._new_data_cb.param.watch(self._on_toggle_new_data, "value")

        self._update_btn = pn.widgets.Button(
            name="Update Model", button_type="primary", max_width=200,
        )
        self._update_status = pn.pane.HTML("")
        self._update_btn.on_click(self._on_update)

        self.state.param.watch(self._on_model_change, "model")
        if self.state.model is not None and self.state.model.is_fitted:
            self._refresh_current_info()

    def _override_row(
        self, label: str, widget: pn.widgets.Widget,
    ) -> tuple[pn.widgets.Checkbox, pn.widgets.Widget]:
        cb = pn.widgets.Checkbox(name=f"Override {label}", value=False, max_width=160)
        widget.disabled = True
        cb.param.watch(lambda e, w=widget: setattr(w, "disabled", not e.new), "value")
        return cb, widget

    def _on_toggle_new_data(self, event: param.parameterized.Event) -> None:
        self._new_data_file.disabled = not event.new
        self._new_data_dataset_sel.disabled = not event.new

    def _on_model_change(self, event: param.parameterized.Event) -> None:
        self._refresh_current_info()

    def _refresh_current_info(self) -> None:
        model = self.state.model
        if model is None or not model.is_fitted:
            self._current_info.object = _pending_box(
                "No fitted model yet — fit one in the Modeling tab first."
            )
            return
        cfg = model.config
        self._current_info.object = (
            f'<div style="background:#f4f4f4;border-left:4px solid #0072B2;'
            f'padding:10px 16px;border-radius:6px;font-family:monospace;'
            f'font-size:0.95em">'
            f'<b>Formula:</b> {model.formula}<br>'
            f'<b>Family:</b> {model.family} &nbsp;|&nbsp; '
            f'<b>draws:</b> {cfg.draws} &nbsp;|&nbsp; '
            f'<b>tune:</b> {cfg.tune} &nbsp;|&nbsp; '
            f'<b>chains:</b> {cfg.chains} &nbsp;|&nbsp; '
            f'<b>cores:</b> {cfg.cores} &nbsp;|&nbsp; '
            f'<b>target_accept:</b> {cfg.target_accept} &nbsp;|&nbsp; '
            f'<b>rows:</b> {len(model.data)}'
            f'</div>'
        )

    # Update Model — update_model() only, never a fresh hbm_<family>() build

    def _collect_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if self._draws_cb.value:
            overrides["draws"] = int(self._draws_in.value)
        if self._tune_cb.value:
            overrides["tune"] = int(self._tune_in.value)
        if self._chains_cb.value:
            overrides["chains"] = int(self._chains_in.value)
        if self._cores_cb.value:
            overrides["cores"] = int(self._cores_in.value)
        if self._target_accept_cb.value:
            overrides["target_accept"] = float(self._target_accept_in.value)
        if self._seed_cb.value:
            overrides["random_seed"] = int(self._seed_in.value)
        return overrides

    def _resolve_new_data(self) -> pd.DataFrame | None:
        """Replacement data if `_new_data_cb` is checked, else `None`.

        Raises on a checked-but-empty selection, or a CSV that fails to
        parse — the caller turns that into an error box instead of
        silently refitting on the old data.
        """
        if not self._new_data_cb.value:
            return None
        if self._new_data_file.value:
            return pd.read_csv(io.BytesIO(self._new_data_file.value))
        if self._new_data_dataset_sel.value:
            return load_dataset(self._new_data_dataset_sel.value)
        raise ValueError(
            "'Refit on different data' is checked, but no CSV was uploaded and "
            "no built-in dataset was selected."
        )

    async def _on_update(self, event: Any) -> None:
        if self._update_btn.loading:  
            return

        model: BaseModel | None = getattr(self.state, "model", None)
        if model is None or not model.is_fitted:
            self._update_status.object = _error_box(
                "No fitted model", "Fit a model in the Modeling tab first."
            )
            return

        try:
            new_data = self._resolve_new_data()
        except Exception as exc:
            title, body = _describe_error(exc)
            self._update_status.object = _error_box(title, body)
            return

        overrides = self._collect_overrides()

        self._update_btn.loading = self._update_btn.disabled = True
        self._update_status.object = _info_box("Refitting model…")
        try:
            loop = asyncio.get_running_loop()
            fitted = await loop.run_in_executor(
                None, self._update_blocking, model, new_data, overrides
            )
        except Exception as exc:
            logger.exception("UpdateModelTab refit failed")
            title, body = _describe_error(exc)
            self._update_status.object = _error_box(title, body)
            return
        finally:
            self._update_btn.loading = self._update_btn.disabled = False

        self.state.model = fitted
        self.state.param.trigger("model")
        if new_data is not None:
            self.state.data = new_data

        self._update_status.object = _success_box(
            "Model refit complete. Open the <b>Results</b> tab again to see the "
            "updated convergence diagnostics and SAE estimates."
        )

    @staticmethod
    def _update_blocking(
        model: BaseModel, new_data: pd.DataFrame | None, overrides: dict[str, Any],
    ) -> BaseModel:
        """Synchronous, Panel-free — runs in the executor thread.

        The single tier-3-adjacent entry point for this tab: always
        `update_model()`, never a fresh `hbm_<family>()` build.
        """
        update_model(model, new_data=new_data, **overrides)
        return model

    def panel(self) -> pn.Column:
        """Return the Panel layout for this tab.

        Returns:
            A ``panel.Column`` with the current fitted model's summary,
            optional sampler overrides, an optional replacement-data
            picker, and the **Update Model** button.
        """
        overview = pn.pane.Markdown(
            "Refit the current model with updated MCMC settings without changing the model specification. The formula, family, link, and grouping remain unchanged. Adjust the available sampler settings as needed, then refit the model and update the results."
        )

        sampler_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("**Sampler Overrides** (optional)", margin=(4, 0, 4, 0)),
                pn.FlexBox(self._draws_cb, self._draws_in),
                pn.FlexBox(self._tune_cb, self._tune_in),
                pn.FlexBox(self._chains_cb, self._chains_in),
                pn.FlexBox(self._cores_cb, self._cores_in),
                pn.FlexBox(self._target_accept_cb, self._target_accept_in),
                pn.FlexBox(self._seed_cb, self._seed_in),
            ),
            title="Sampler Settings",
            margin=10,
        )

        data_card = pn.Card(
            pn.Column(
                self._new_data_cb,
                pn.FlexBox(self._new_data_file, self._new_data_dataset_sel),
            ),
            title="Replacement Data",
            margin=10,
        )

        return pn.Column(
            pn.Card(overview, title="Overview", margin=10),
            pn.Card(self._current_info, title="Current Fitted Model", margin=10),
            sampler_card,
            data_card,
            pn.Row(self._update_btn),
            self._update_status,
        )

    def __repr__(self) -> str:
        model = getattr(self.state, "model", None)
        fitted = bool(model is not None and getattr(model, "is_fitted", False))
        return f"UpdateModelTab(has_fitted_model={fitted})"