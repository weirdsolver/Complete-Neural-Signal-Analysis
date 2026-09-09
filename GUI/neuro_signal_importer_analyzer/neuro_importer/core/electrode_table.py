from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def make_electrode_table(
    labels: Iterable[str],
    *,
    device: str | None = None,
    x: Iterable[float] | None = None,
    y: Iterable[float] | None = None,
    z: Iterable[float] | None = None,
    extra: dict[str, Iterable[Any] | Any] | None = None,
) -> pd.DataFrame:
    """Create an electrode/channel geometry table.

    Geometry fields are optional because many exports expose channel numbers but
    not physical coordinates. The table is still useful as a stable place for
    MEA/well/organoid/device-channel metadata.
    """
    names = list(labels)
    n = len(names)
    data: dict[str, Any] = {
        "index": list(range(n)),
        "name": [str(v) for v in names],
    }
    if device is not None:
        data["device"] = [device] * n
    if x is not None:
        vals = list(x)
        data["x"] = vals[:n] + [None] * max(0, n - len(vals))
    if y is not None:
        vals = list(y)
        data["y"] = vals[:n] + [None] * max(0, n - len(vals))
    if z is not None:
        vals = list(z)
        data["z"] = vals[:n] + [None] * max(0, n - len(vals))
    if extra:
        for key, value in extra.items():
            if isinstance(value, (list, tuple)):
                vals = list(value)
                data[str(key)] = vals[:n] + [None] * max(0, n - len(vals))
            else:
                data[str(key)] = [value] * n
    return pd.DataFrame(data)
