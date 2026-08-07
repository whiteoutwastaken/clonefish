"""
Standalone smoke test for RVCStream.

Run this directly (`python -m tools.test_rvc`) to sanity-check a
model loads and process() runs, without spinning up the full ZMQ
pipeline. This does NOT test conversion quality - silence in only
confirms the pipeline runs without crashing and that output values
land in a sane range, not that the voice sounds right.
"""

import time

import numpy as np

from tools.rvc import RVCStream


if __name__ == "__main__":

    rvc = RVCStream(
        "shylily"
    )

    audio = np.zeros(
        15360,
        dtype=np.float32
    )

    NUM_RUNS = 10

    timings_ms = []

    for i in range(NUM_RUNS):

        start = time.perf_counter()

        out = rvc.process(
            audio
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        timings_ms.append(
            elapsed_ms
        )

        print(
            f"run {i}: {elapsed_ms:.1f} ms"
        )

    print()

    print(
        "Output:",
        out.shape,
        out.dtype,
        "peak:",
        np.abs(out).max() if out.size else 0
    )

    print()

    # First call pays for CUDA context init, cuDNN kernel
    # autotuning, and moving RMVPE onto the GPU for the first
    # time - none of that repeats on later calls, so it's
    # excluded here to show the number that actually matters:
    # steady-state per-frame cost against the 50ms hop budget.
    steady_state = timings_ms[1:]

    avg_ms = sum(steady_state) / len(steady_state)

    print(
        f"run 0 (cold): {timings_ms[0]:.1f} ms"
    )

    print(
        f"runs 1-{NUM_RUNS - 1} (warm) avg: {avg_ms:.1f} ms "
        f"(hop budget: 50 ms)"
    )