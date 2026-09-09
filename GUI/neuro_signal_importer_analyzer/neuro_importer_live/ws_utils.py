from __future__ import annotations

import json
import struct

import numpy as np


def pack_binary_float32_frame(header: dict, payload: np.ndarray) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes)
    payload_offset = 8 + header_len
    pad = (4 - (payload_offset % 4)) % 4
    payload_offset += pad
    return struct.pack("<II", header_len, payload_offset) + header_bytes + (b"\x00" * pad) + np.ascontiguousarray(payload, dtype=np.float32).tobytes(order="C")
