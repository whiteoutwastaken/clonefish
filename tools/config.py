"""Shared tunables for the RVC voice-changer pipeline."""

import os

# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "egirl"
PITCH_SHIFT_SEMITONES = 15   # try 3-7 for a natural-sounding raise

# --------------------------------------------------
# Audio devices
# --------------------------------------------------

SAMPLE_RATE = 48000
INPUT_DEVICE = None   # your mic, sounddevice index. None = OS default.
OUTPUT_DEVICE = None  # your headset, sounddevice index. None = OS default.


def get_input_channels(device=INPUT_DEVICE):
    """Query the actual channel count the device supports for input,
    instead of assuming 1. Falls back to the OS default input device
    when device is None.

    sounddevice is imported lazily here (not at module scope) because
    config.py is shared across conda envs - rvc.py's env has no
    sounddevice installed and never needs this function."""

    import sounddevice as sd

    info = sd.query_devices(device, kind="input")
    channels = info["max_input_channels"]

    if channels < 1:
        raise RuntimeError(
            f"Selected input device ({device!r}) reports "
            f"{channels} input channels: {info['name']!r}"
        )

    return channels


def get_output_channels(device=OUTPUT_DEVICE):
    """Query the actual channel count the device supports for output,
    instead of assuming 1. Falls back to the OS default output device
    when device is None.

    sounddevice is imported lazily here for the same reason as
    get_input_channels above."""

    import sounddevice as sd

    info = sd.query_devices(device, kind="output")
    channels = info["max_output_channels"]

    if channels < 1:
        raise RuntimeError(
            f"Selected output device ({device!r}) reports "
            f"{channels} output channels: {info['name']!r}"
        )

    return channels


# --------------------------------------------------
# Windowing
#
# Each packet sent to RVC is WINDOW_MS long. Only the middle STEP_MS
# is actually new/unheard audio - CONTEXT_MS on each side is real
# context that helps HuBERT/RMVPE quality at the boundaries, at the
# cost of latency: nothing can be sent until CONTEXT_MS of audio
# *after* the target slice has been captured. Total pipeline latency
# is roughly STEP_MS + CONTEXT_MS + RVC compute time.
#
# Throughput constraint: process_time(WINDOW_MS) must stay below
# STEP_MS or the worker falls behind. Watch the
# "[rvc] process() took Nms" log against STEP_MS when tuning.
# --------------------------------------------------

WINDOW_MS = 620
STEP_MS = 500
CONTEXT_MS = (WINDOW_MS - STEP_MS) / 2  # 60ms each side, at these defaults

# --------------------------------------------------
# Crossfade
#
# FADE_MS of extra audio is extracted on each side of the STEP_MS
# core (in addition to CONTEXT_MS) so output.py can crossfade the
# tail of one packet into the head of the next instead of hard-
# cutting between independently-generated conversions.
# --------------------------------------------------

FADE_MS = 20

assert FADE_MS <= CONTEXT_MS, "FADE_MS must fit inside CONTEXT_MS"
assert 2 * FADE_MS < STEP_MS, "2*FADE_MS must be smaller than STEP_MS"


def _samples(ms, rate=SAMPLE_RATE):
    return int(rate * ms / 1000)


WINDOW_SAMPLES = _samples(WINDOW_MS)
STEP_SAMPLES = _samples(STEP_MS)
FADE_SAMPLES = _samples(FADE_MS)

# --------------------------------------------------
# Networking - ports are chosen dynamically by main.py at launch
# (to sidestep stale binds from a crashed previous run) and passed
# to each subprocess via these env vars. Defaults below only apply
# when running a stage standalone (e.g. `python -m tools.input`).
# --------------------------------------------------

INPUT_TO_RVC_PORT_ENV = "RVC_PORT_IN"
RVC_TO_OUTPUT_PORT_ENV = "RVC_PORT_OUT"

DEFAULT_INPUT_TO_RVC_PORT = 5555
DEFAULT_RVC_TO_OUTPUT_PORT = 5556


def get_port(env_name, default):
    return int(os.environ.get(env_name, default))

# --------------------------------------------------
# Silence gate
#
# Below this peak amplitude (raw 48kHz), a window is treated as
# silence and skipped. HANGOVER_PACKETS controls how many
# subsequent windows stay "forced open" after a voiced window,
# even if they individually measure below threshold - this stops
# natural trailing-off speech (fading cadence) from getting cut
# off mid-decay.
# --------------------------------------------------

SILENCE_THRESHOLD = 0.02
HANGOVER_PACKETS = 1