"""Centralized logging for ``hbsaemp``. No ``print()`` allowed in library code.

R→Python mapping:  message()→logger.info,  warning()→logger.warning,
                   stop()→raise,  cat()→logger.debug

Default: NullHandler (silent). Call :func:`configure_logging` to opt in.
"""

from __future__ import annotations

import logging
from typing import Union

__all__ = ["PACKAGE_LOGGER_NAME", "DEFAULT_LOG_FORMAT", "DEFAULT_DATE_FORMAT",
           "get_logger", "configure_logging"]

PACKAGE_LOGGER_NAME: str = "hbsaemp"
DEFAULT_LOG_FORMAT: str = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger. Call with ``__name__`` in every module.

    Args:
        name: Module name. ``None`` → root ``hbsaemp`` logger.
            Names not starting with ``"hbsaemp."`` are auto-prefixed.

    Returns:
        A ready :class:`logging.Logger`.

    Raises:
        TypeError: If *name* is not ``str`` or ``None``.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError(f"`name` must be str or None, got {type(name).__name__!r}")
    if not name or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if not name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        name = f"{PACKAGE_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    *,
    fmt: str | None = None,
    datefmt: str | None = None,
    propagate: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Attach a StreamHandler to the root logger (opt-in for interactive use).

    Idempotent when ``force=False``.

    Args:
        level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` or int.
        fmt: Override :data:`DEFAULT_LOG_FORMAT`.
        datefmt: Override :data:`DEFAULT_DATE_FORMAT`.
        propagate: Propagate to Python root logger.
        force: Remove existing handlers first.

    Returns:
        The configured root :class:`logging.Logger`.

    Raises:
        TypeError: If *level* is not ``int`` or ``str``.
        ValueError: If *level* string is not a valid level name.
    """
    if isinstance(level, str):
        numeric: int = logging.getLevelName(level.upper())
        if not isinstance(numeric, int):
            raise ValueError(
                f"Unknown level {level!r}. Use DEBUG/INFO/WARNING/ERROR/CRITICAL."
            )
        level = numeric
    elif not isinstance(level, int):
        raise TypeError(f"`level` must be int or str, got {type(level).__name__!r}")

    root = logging.getLogger(PACKAGE_LOGGER_NAME)

    if force:
        for h in list(root.handlers):
            if not isinstance(h, logging.NullHandler):
                root.removeHandler(h)

    if not any(isinstance(h, logging.StreamHandler)
               and not isinstance(h, logging.NullHandler)
               for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt=fmt or DEFAULT_LOG_FORMAT, datefmt=datefmt or DEFAULT_DATE_FORMAT
        ))
        root.addHandler(handler)

    root.setLevel(level)
    root.propagate = propagate
    return root
