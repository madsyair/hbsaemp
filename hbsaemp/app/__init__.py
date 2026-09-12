from __future__ import annotations
from hbsaemp._logging import get_logger
from hbsaemp.app._app import App
from hbsaemp.app._config import AppConfig, DEFAULT_APP_CONFIG

logger = get_logger(__name__)
__all__: list[str] = ["launch_app", "App", "AppConfig", "DEFAULT_APP_CONFIG"]


def launch_app(
    *,
    port: int | None = None,
    title: str | None = None,
    open_browser: bool = True,
    app_config: AppConfig | None = None,
) -> None:
    """Launch the hbsaemp web dashboard

    Args:
        port: Web server port. Default 8080.
        title: Browser tab title.
        open_browser: Open browser on launch. Default True.
        app_config: Full :class:`AppConfig` — overrides other kwargs.

    Raises:
        ValueError: If the resulting :class:`AppConfig` is invalid — see
            :class:`AppConfig` for the specific checks.

    Examples:
        >>> import hbsaemp
        >>> hbsaemp.launch_app()                    # default
        >>> hbsaemp.launch_app(port=8888)           # custom port
        >>> hbsaemp.launch_app(open_browser=False)  # headless
    """
    if app_config is None:
        app_config = AppConfig(
            port=port or DEFAULT_APP_CONFIG.port,
            title=title or DEFAULT_APP_CONFIG.title,
            open_browser=open_browser,
        )
    logger.info("launch_app: port=%d", app_config.port)
    App(app_config=app_config).serve()