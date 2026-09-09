from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


EEG_10_10_COMMON_32 = [
    "Fp1", "Fpz", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO3", "POz", "PO4",
    "O1", "Oz", "O2",
]

EEG_10_20_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

AUX_PREFIXES = ("AUX", "RESP", "TRIG", "STIM", "EOG", "ECG", "EMG", "REF", "GND", "BIP")


@dataclass
class ChannelManifest:
    n_channels: int
    channel_names: list[str]
    channel_types: list[str] = field(default_factory=list)
    channel_namespace: str = "generated_numeric"
    modality: str = "continuous_neural"
    units: Optional[str] = None
    sample_rate_hz: Optional[float] = None
    groups: dict[str, list[str]] = field(default_factory=dict)
    geometry: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError("ChannelManifest requires at least one channel")
        if len(self.channel_names) != self.n_channels:
            raise ValueError(
                f"channel_names length {len(self.channel_names)} does not match n_channels {self.n_channels}"
            )
        if not self.channel_types:
            self.channel_types = [infer_channel_type(name, self.modality) for name in self.channel_names]
        elif len(self.channel_types) != self.n_channels:
            raise ValueError(
                f"channel_types length {len(self.channel_types)} does not match n_channels {self.n_channels}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any], n_channels: Optional[int] = None) -> "ChannelManifest":
        names = data.get("channel_names") or data.get("names") or []
        if n_channels is None:
            n_channels = int(data.get("n_channels") or len(names))
        if not names:
            names = generated_channel_names(n_channels)
        return cls(
            n_channels=int(n_channels),
            channel_names=list(map(str, names)),
            channel_types=list(map(str, data.get("channel_types") or data.get("types") or [])),
            channel_namespace=str(data.get("channel_namespace") or data.get("namespace") or "generated_numeric"),
            modality=str(data.get("modality") or "continuous_neural"),
            units=data.get("units"),
            sample_rate_hz=_maybe_float(data.get("sample_rate_hz") or data.get("fs") or data.get("sampling_rate")),
            groups={str(k): list(map(str, v)) for k, v in (data.get("groups") or {}).items()},
            geometry=data.get("geometry") or {},
            extra=data.get("extra") or {},
        )

    @classmethod
    def from_json(cls, text: str, n_channels: Optional[int] = None) -> "ChannelManifest":
        return cls.from_dict(json.loads(text), n_channels=n_channels)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def generated_channel_names(n_channels: int, prefix: str = "ch") -> list[str]:
    width = max(3, len(str(max(n_channels - 1, 0))))
    return [f"{prefix}_{i:0{width}d}" for i in range(n_channels)]


def infer_channel_type(name: str, modality: str = "continuous_neural") -> str:
    upper = str(name).upper()
    if upper.startswith(AUX_PREFIXES):
        return upper.split("_")[0]
    if modality.upper() in {"EEG", "ECOG", "SEEG", "MEA"}:
        return modality.upper()
    if upper.startswith(("E", "CH", "CHANNEL", "ELECTRODE", "MEA")):
        return "MEA" if "MEA" in upper or "ELECTRODE" in upper else "NEURAL"
    return "NEURAL"


def _groups_for_eeg(names: list[str]) -> dict[str, list[str]]:
    name_set = set(names)
    groups = {
        "frontal": [n for n in names if n.upper().startswith(("FP", "AF", "F"))],
        "central": [n for n in names if n.upper().startswith(("FC", "C", "CP"))],
        "posterior": [n for n in names if n.upper().startswith(("P", "PO", "O"))],
        "left": [n for n in names if n[-1:].isdigit() and int(n[-1]) % 2 == 1],
        "right": [n for n in names if n[-1:].isdigit() and int(n[-1]) % 2 == 0],
        "midline": [n for n in names if n.endswith("z") or n.endswith("Z")],
    }
    return {k: [n for n in v if n in name_set] for k, v in groups.items() if v}


def finalspark_32_manifest(sample_rate_hz: Optional[float] = None, units: Optional[str] = None) -> ChannelManifest:
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    geometry: dict[str, dict[str, int | str]] = {}
    for organoid in range(4):
        group: list[str] = []
        for electrode in range(8):
            name = f"mea0_organoid{organoid}_e{electrode}"
            names.append(name)
            group.append(name)
            geometry[name] = {"mea": 0, "organoid": organoid, "electrode": electrode}
        groups[f"organoid_{organoid}"] = group
    groups["mea_0"] = names.copy()
    return ChannelManifest(
        n_channels=32,
        channel_names=names,
        channel_types=["MEA"] * 32,
        channel_namespace="FinalSpark_4Organoids_8ElectrodesEach",
        modality="MEA",
        units=units,
        sample_rate_hz=sample_rate_hz,
        groups=groups,
        geometry=geometry,
    )


def profile_manifest(profile: str, n_channels: int, sample_rate_hz: Optional[float] = None, units: Optional[str] = None) -> ChannelManifest:
    key = (profile or "auto").lower().replace("-", "_")
    if key in {"finalspark", "finalspark_32", "finalspark_live_mea"}:
        if n_channels != 32:
            # FinalSpark platform profile is 32 electrodes per MEA, but do not block files.
            names = generated_channel_names(n_channels, prefix="electrode")
            return ChannelManifest(n_channels, names, ["MEA"] * n_channels, "FinalSpark_like_generated", "MEA", units, sample_rate_hz)
        return finalspark_32_manifest(sample_rate_hz, units)
    if key in {"eeg_10_10_32", "eeg32", "32_eeg"} and n_channels == 32:
        return ChannelManifest(n_channels, EEG_10_10_COMMON_32.copy(), ["EEG"] * 32, "EEG_10_10_common_32", "EEG", units, sample_rate_hz, _groups_for_eeg(EEG_10_10_COMMON_32))
    if key in {"eeg_10_20", "eeg_10_20_19"} and n_channels == 19:
        return ChannelManifest(n_channels, EEG_10_20_19.copy(), ["EEG"] * 19, "EEG_10_20_19", "EEG", units, sample_rate_hz, _groups_for_eeg(EEG_10_20_19))
    names = generated_channel_names(n_channels)
    modality = "MEA" if "mea" in key else "continuous_neural"
    return ChannelManifest(n_channels, names, [infer_channel_type(n, modality) for n in names], "generated_numeric", modality, units, sample_rate_hz)


def read_channels_csv(path: Path, n_channels: Optional[int] = None, sample_rate_hz: Optional[float] = None, units: Optional[str] = None) -> Optional[ChannelManifest]:
    path = Path(path)
    if not path.exists():
        return None
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
    if not rows:
        return None
    names: list[str] = []
    types: list[str] = []
    geometry: dict[str, dict[str, float | int | str]] = {}
    for i, row in enumerate(rows):
        name = row.get("name") or row.get("channel_name") or row.get("label") or row.get("channel") or row.get("electrode") or f"ch_{i:03d}"
        names.append(str(name))
        types.append(row.get("type") or row.get("channel_type") or row.get("kind") or "NEURAL")
        geo: dict[str, float | int | str] = {}
        for k in ("x", "y", "z", "row", "column", "well", "organoid", "mea", "electrode"):
            if k in row and row[k] != "":
                geo[k] = _number_or_str(row[k])
        if geo:
            geometry[str(name)] = geo
    if n_channels is not None and len(names) != int(n_channels):
        names = (names + generated_channel_names(int(n_channels))[len(names):])[: int(n_channels)]
        types = (types + ["NEURAL"] * int(n_channels))[: int(n_channels)]
    n = int(n_channels or len(names))
    return ChannelManifest(
        n_channels=n,
        channel_names=names[:n],
        channel_types=types[:n],
        channel_namespace="channels_csv",
        modality=_infer_modality_from_types(types),
        units=units,
        sample_rate_hz=sample_rate_hz,
        groups={},
        geometry=geometry,
    )


def _number_or_str(value: str) -> float | int | str:
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except Exception:
        return value


def _infer_modality_from_types(types: Iterable[str]) -> str:
    upper = {str(t).upper() for t in types}
    for candidate in ("EEG", "ECOG", "SEEG", "MEA"):
        if candidate in upper:
            return candidate
    return "continuous_neural"


def read_metadata_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_manifest(
    n_channels: int,
    *,
    channels_csv: Optional[Path] = None,
    metadata_json: Optional[Path] = None,
    profile: str = "auto",
    sample_rate_hz: Optional[float] = None,
    units: Optional[str] = None,
) -> ChannelManifest:
    metadata = read_metadata_json(metadata_json)
    if sample_rate_hz is None:
        sample_rate_hz = _maybe_float(
            metadata.get("sample_rate_hz")
            or metadata.get("sampling_rate_hz")
            or metadata.get("sampling_rate")
            or metadata.get("fs")
        )
    if units is None:
        units = metadata.get("units") or metadata.get("signal_units") or metadata.get("target_units")

    manifest_data = metadata.get("channel_manifest") or metadata.get("channels_manifest")
    if isinstance(manifest_data, dict):
        manifest = ChannelManifest.from_dict(manifest_data, n_channels=n_channels)
        manifest.sample_rate_hz = manifest.sample_rate_hz or sample_rate_hz
        manifest.units = manifest.units or units
        return manifest

    if channels_csv:
        manifest = read_channels_csv(Path(channels_csv), n_channels=n_channels, sample_rate_hz=sample_rate_hz, units=units)
        if manifest is not None:
            return manifest

    if profile == "auto":
        if n_channels == 32:
            return profile_manifest("eeg_10_10_32", n_channels, sample_rate_hz, units)
        return profile_manifest("generated_numeric", n_channels, sample_rate_hz, units)
    return profile_manifest(profile, n_channels, sample_rate_hz, units)
