"""NeuroMouse integration layer for the Neuro Signal Importer.

This package produces NeuroMouse-compatible static datasets, comparison packs,
and live replay sample frames from the importer canonical output format.
"""

from .static_dataset_builder import build_speedmouse_dataset, write_speedmouse_dataset
from .comparison_packager import build_speedmouse_comparison_pack

__all__ = [
    "build_speedmouse_dataset",
    "write_speedmouse_dataset",
    "build_speedmouse_comparison_pack",
]
