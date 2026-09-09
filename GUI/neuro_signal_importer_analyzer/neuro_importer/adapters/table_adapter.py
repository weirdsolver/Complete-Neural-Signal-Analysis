from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_importer.adapters.base import AdapterScore, BaseAdapter
from neuro_importer.adapters.common import apply_aux_filter, make_channel_table, synthesize_or_validate_time
from neuro_importer.core.quality import QualityReport
from neuro_importer.core.recording import Recording


class TableAdapter(BaseAdapter):
    """Adapter for CSV/TSV/Excel tables where rows are samples and columns are channels."""

    name = "table_signal"
    time_column_candidates = ("time", "times", "t", "timestamp", "timestamps", "seconds", "sec")

    def score(self, raw: Any) -> AdapterScore:
        df = self._extract_dataframe(raw)
        if df is None:
            return AdapterScore(self.name, 0.0, ["no pandas DataFrame found"])
        numeric_cols = self._numeric_signal_columns(df)
        reasons = [f"table found with {len(df)} rows and {len(df.columns)} columns"]
        confidence = 0.0
        if len(numeric_cols) >= 1 and len(df) >= 2:
            confidence += 0.55
            reasons.append(f"{len(numeric_cols)} numeric signal-like columns found")
        if self._find_time_column(df) is not None:
            confidence += 0.20
            reasons.append("time-like column found")
        if len(numeric_cols) >= 4:
            confidence += 0.10
        return AdapterScore(self.name, min(confidence, 0.85), reasons)

    def convert(
        self,
        raw: Any,
        *,
        source_path: str | Path | None = None,
        subject: int | float | str | None = None,
        session: int | float | str | None = None,
        include_aux: bool = False,
        sampling_rate: float | None = None,
        time_column: str | None = None,
        **_: Any,
    ) -> Recording:
        q = QualityReport(adapter=self.name)
        score = self.score(raw)
        q.confidence = score.confidence
        for reason in score.reasons:
            q.add_info(reason)
        df = self._extract_dataframe(raw)
        if df is None:
            q.add_error("No table/DataFrame could be extracted.")
            raise ValueError(q.errors[-1])

        tc = time_column or self._find_time_column(df)
        time = None
        if tc is not None and tc in df.columns:
            time = pd.to_numeric(df[tc], errors="coerce").to_numpy(dtype=float)
            if np.isnan(time).all():
                time = None
            else:
                q.add_info(f"Using {tc!r} as time column.")

        signal_cols = self._numeric_signal_columns(df, exclude={tc} if tc else set())
        if not signal_cols:
            q.add_error("No numeric signal columns found in table.")
            raise ValueError(q.errors[-1])
        signal = df[signal_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if sampling_rate is None and time is not None and len(time) > 1:
            diffs = np.diff(time[np.isfinite(time)])
            diffs = diffs[diffs > 0]
            if diffs.size:
                sampling_rate = float(1.0 / np.median(diffs))
                q.add_info("Inferred sampling rate from median time step.")
        channels = make_channel_table(signal_cols, signal.shape[1], quality=q)
        signal, channels = apply_aux_filter(signal, channels, include_aux=include_aux, quality=q)
        time = synthesize_or_validate_time(time, signal.shape[0], sampling_rate, q)

        metadata = {
            "format": "Table",
            "source_path": str(source_path) if source_path is not None else None,
            "sampling_rate": sampling_rate,
            "subject": subject,
            "session": session,
            "time_column": tc,
            "signal_columns": signal_cols,
            "canonical_signal_shape": tuple(int(x) for x in signal.shape),
        }
        return Recording(signal=signal, sampling_rate=sampling_rate, time=time, channels=channels, metadata=metadata, quality=q, source_path=str(source_path) if source_path is not None else None)

    def _extract_dataframe(self, raw: Any) -> pd.DataFrame | None:
        if isinstance(raw, pd.DataFrame):
            return raw
        if isinstance(raw, dict) and isinstance(raw.get("dataframe"), pd.DataFrame):
            return raw["dataframe"]
        return None

    def _find_time_column(self, df: pd.DataFrame) -> str | None:
        lower_map = {str(c).lower().strip(): c for c in df.columns}
        for name in self.time_column_candidates:
            if name in lower_map:
                return lower_map[name]
        for c in df.columns:
            lc = str(c).lower().strip()
            if "time" in lc or "timestamp" in lc:
                return c
        return None

    def _numeric_signal_columns(self, df: pd.DataFrame, exclude: set[str | None] | None = None) -> list[str]:
        exclude = exclude or set()
        out: list[str] = []
        for c in df.columns:
            if c in exclude:
                continue
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() >= max(2, int(0.5 * len(series))):
                out.append(c)
        return out
