"""Stands in for arecord in tests: streams 16 kHz mono audio in real time, with background noise.

Audio files (raw 16-bit, 16 kHz, mono) put in the spool folder are played in name order, then deleted;
between them it streams background noise only. Use it as stt_node's recorder:

    recorder: "python3 .../fake_microphone.py --spool /tmp/mic --noise 0.01"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

SAMPLE_RATE = 16000
FRAME = 480  # 30 ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spool", required=True, help="folder to take audio files from")
    parser.add_argument("--noise", type=float, default=0.0015, help="rms of the background noise")
    args, _arecord_arguments = parser.parse_known_args()
    os.makedirs(args.spool, exist_ok=True)
    rng = np.random.default_rng(0)
    out = sys.stdout.buffer
    pending = np.zeros(0, dtype=np.float32)
    started, sent = time.monotonic(), 0
    while True:
        if not len(pending):
            ready = sorted(name for name in os.listdir(args.spool) if name.endswith(".raw"))
            if ready:
                path = os.path.join(args.spool, ready[0])
                pending = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768.0
                os.remove(path)
        frame = np.zeros(FRAME, dtype=np.float32)
        if len(pending):
            chunk, pending = pending[:FRAME], pending[FRAME:]
            frame[: len(chunk)] = chunk
        smooth = np.convolve(rng.standard_normal(FRAME + 7), np.ones(8) / 8, mode="valid")
        frame += smooth / np.sqrt(np.mean(smooth**2)) * args.noise
        out.write((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())
        out.flush()
        sent += FRAME
        time.sleep(max(0.0, started + sent / SAMPLE_RATE - time.monotonic()))  # real time


if __name__ == "__main__":
    main()
