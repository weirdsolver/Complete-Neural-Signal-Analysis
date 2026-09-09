"""NeuroMouse integration layer for the Neuro Signal Importer.

This package produces NeuroMouse-compatible static datasets, comparison packs,
and live replay sample frames from the importer canonical output format.
"""

from .static_dataset_builder import build_speedmouse_dataset, write_speedmouse_dataset
from .comparison_packager import build_neuromouse_comparison_pack

# Preferred NeuroMouse API names.
build_neuromouse_dataset = build_speedmouse_dataset
write_neuromouse_dataset = write_speedmouse_dataset

# Legacy internal names kept so older code/tests still import successfully.
build_speedmouse_comparison_pack = build_neuromouse_comparison_pack

__all__ = [
    "build_neuromouse_dataset",
    "write_neuromouse_dataset",
    "build_neuromouse_comparison_pack",
    "build_speedmouse_dataset",
    "write_speedmouse_dataset",
    "build_speedmouse_comparison_pack",
]
