from __future__ import annotations

import importlib
import os
from importlib.metadata import entry_points
from typing import Any


def _instantiate(obj: Any) -> Any:
    return obj() if isinstance(obj, type) else obj


def load_plugin_adapters() -> list[Any]:
    """Load optional adapter plugins.

    Supported mechanisms:
    1. Python package entry points under group ``neuro_importer.adapters``.
    2. Environment variable ``NEURO_IMPORTER_ADAPTERS`` with comma-separated
       ``module:ClassName`` references.
    """
    adapters: list[Any] = []
    try:
        eps = entry_points()
        selected = eps.select(group='neuro_importer.adapters') if hasattr(eps, 'select') else eps.get('neuro_importer.adapters', [])
        for ep in selected:
            try:
                adapters.append(_instantiate(ep.load()))
            except Exception:
                continue
    except Exception:
        pass
    env = os.environ.get('NEURO_IMPORTER_ADAPTERS', '')
    for item in [x.strip() for x in env.split(',') if x.strip()]:
        try:
            module_name, attr = item.split(':', 1)
            module = importlib.import_module(module_name)
            adapters.append(_instantiate(getattr(module, attr)))
        except Exception:
            continue
    return adapters
