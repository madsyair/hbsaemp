"""GUI application configuration for hbsaemp
"""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["AppConfig", "DEFAULT_APP_CONFIG"]


@dataclass
class AppConfig:
    """Configuration for the hbsaemp web dashboard.

    Args:
        title: Browser tab and header title. Default ``"HBSAEMP APP"``.
        port: Port on which the server listens. Default ``8080``.
        open_browser: Automatically open the browser on launch.
            Default ``True``.
        theme: Visual design for the dashboard chrome. One of ``"bootstrap"``
            (default, light) or ``"dark"``. Read by
            :meth:`~hbsaemp.app._app.App.view`, which maps it to a Panel
            ``Theme`` class (``DarkTheme`` for ``"dark"``, ``DefaultTheme``
            otherwise — unrecognised values fall back to the default rather
            than raising, since this only affects appearance).
        max_upload_mb: Maximum file upload size in megabytes. Enforced by
            :class:`~hbsaemp.app.tabs.data_tab.DataTab` when a CSV is
            uploaded; larger files are rejected with an in-UI error instead
            of being read into memory. Default 50.
        show_sidebar: Whether to render the sidebar (workflow summary).
            Read by :meth:`~hbsaemp.app._app.App.view`. Default ``True``.
        log_level: Logging level applied to the ``hbsaemp`` logger via
            :func:`~hbsaemp._logging.configure_logging` when the app starts.
            Default ``"WARNING"``.
        accent: Accent color (hex) used by the ``FastListTemplate``
            dashboard header/highlights. Default ``"#A01346"``.

    Raises:
        ValueError: If ``port`` is outside 1-65535, ``max_upload_mb`` is
            less than 1, ``log_level`` is not one of the standard logging
            level names, or ``theme`` is not ``"bootstrap"`` or ``"dark"``.
            Checked once, at construction time.
    """

    title: str = "HBSAEMP APP"
    port: int = 8080
    open_browser: bool = True
    theme: str = "bootstrap"
    max_upload_mb: int = 50
    show_sidebar: bool = True
    log_level: str = "WARNING"
    accent: str = "#A01346"

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"`port` must be 1-65535, got {self.port}")
        if self.max_upload_mb < 1:
            raise ValueError(f"`max_upload_mb` must be >= 1, got {self.max_upload_mb}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid `log_level`: {self.log_level!r}")
        if self.theme not in ("bootstrap", "dark"):
            raise ValueError(
                f"Invalid `theme`: {self.theme!r}. Use 'bootstrap' or 'dark'."
            )

    def __repr__(self) -> str:
        return (
            f"AppConfig(title={self.title!r}, port={self.port}, "
            f"open_browser={self.open_browser}, theme={self.theme!r})"
        )

DEFAULT_APP_CONFIG: AppConfig = AppConfig()