"""Event Window-JEPA.

The public package deliberately exposes only stable, experiment-facing APIs.
"""

from event_window_jepa.models.window_jepa import WindowJEPA, WindowJEPAOutput

__all__ = ["WindowJEPA", "WindowJEPAOutput"]
__version__ = "0.1.0"

