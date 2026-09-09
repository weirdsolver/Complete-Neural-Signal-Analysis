from __future__ import annotations

from pathlib import Path

import numpy as np

from neuro_importer.core.recording import Recording


class NeuralSignalExporter:
    """Export neural signal only, in notebook-friendly forms."""

    def export(self, recording: Recording, output_dir: str | Path) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eeg_df_path = out / "eeg_df.csv"
        neural_npy_path = out / "neural_signal.npy"
        with_time_path = out / "neural_signal_with_time.csv"

        # Match uploaded notebook style: channels first, Time last.
        recording.to_dataframe(time_column="Time", time_last=True).to_csv(eeg_df_path, index=False)

        # Model-friendly matrix: samples × channels.
        np.save(neural_npy_path, recording.signal)

        # Analysis-friendly table: Time first, then channels.
        recording.to_dataframe(time_column="Time", time_last=False).to_csv(with_time_path, index=False)

        return {
            "eeg_df": str(eeg_df_path),
            "neural_signal": str(neural_npy_path),
            "neural_signal_with_time": str(with_time_path),
        }
