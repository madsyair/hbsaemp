"""GUI application configuration for hbsaemp.

:class:`AppConfig` controls how the web application is launched and
styled.  It is separate from :class:`~hbsaemp.models._config.ModelConfig`
(which controls MCMC settings).

Mapping to R hbsaems
---------------------
R's ``run_sae_app()`` has no equivalent config object — the app is
launched with fixed settings.  ``AppConfig`` makes the Python version
more flexible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__: list[str] = ["AppConfig", "DEFAULT_APP_CONFIG"]


@dataclass
class AppConfig:
    """Configuration for the hbsaemp web dashboard.

    Args:
        title: Browser tab and header title. Default ``"hbsaemp — HBSAE"``.
        port: Port on which the server listens. Default ``8080``.
        open_browser: Automatically open the browser on launch.
            Default ``True``.
        theme: Panel/Dash visual theme.  Default ``"bootstrap"``.
        max_upload_mb: Maximum file upload size in megabytes. Default 50.
        show_sidebar: Whether to render a collapsible sidebar with
            navigation. Default ``True``.
        log_level: Logging level for the web server component.
            Default ``"WARNING"``.

    Example:
        >>> cfg = AppConfig(port=8888, title="My SAE App")
        >>> launch_app(app_config=cfg)
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
            raise ValueError(f"`port` must be 1–65535, got {self.port}")
        if self.max_upload_mb < 1:
            raise ValueError(f"`max_upload_mb` must be ≥ 1, got {self.max_upload_mb}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid `log_level`: {self.log_level!r}")

    def __repr__(self) -> str:
        return (
            f"AppConfig(title={self.title!r}, port={self.port}, "
            f"open_browser={self.open_browser}, theme={self.theme!r})"
        )


#: Default app configuration used when ``app_config=None`` is passed to
#: :func:`~hbsaemp.app.launch_app`.
DEFAULT_APP_CONFIG: AppConfig = AppConfig()
