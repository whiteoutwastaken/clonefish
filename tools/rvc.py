"""
Real-time RVC worker.

Input:
    ZeroMQ stream of float32 mono audio arrays (48kHz)

Output:
    ZeroMQ stream of converted float32 audio arrays (48kHz)

Architecture:

input.py
    |
    | tcp://127.0.0.1:5555
    v

rvc.py
    |
    | tcp://127.0.0.1:5556
    v

output.py
"""


from pathlib import Path
import threading
import time

import numpy as np
import librosa
import torch
import zmq


# --------------------------------------------------
# PyTorch 2.6+ compatibility
# --------------------------------------------------

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):

    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _original_torch_load(
        *args,
        **kwargs
    )


torch.load = _patched_torch_load


from rvc_python.infer import RVCInference
from rvc_python.modules.vc.utils import load_hubert



# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    / "models"
)



# --------------------------------------------------
# Model cache
# --------------------------------------------------

_MODEL_CACHE = {}

_MODEL_LOCK = threading.Lock()



def _find_model_files(model_name):

    folder = MODEL_DIR / model_name


    if not folder.exists():

        raise FileNotFoundError(
            f"Missing model folder: {folder}"
        )


    pth = list(
        folder.glob("*.pth")
    )

    index = list(
        folder.glob("*.index")
    )


    if len(pth) != 1:

        raise RuntimeError(
            f"Expected one .pth file, found {len(pth)}"
        )


    if len(index) != 1:

        raise RuntimeError(
            f"Expected one .index file, found {len(index)}"
        )


    return (
        str(pth[0]),
        str(index[0])
    )



def _load_model(model_name):

    with _MODEL_LOCK:


        if model_name in _MODEL_CACHE:

            return _MODEL_CACHE[model_name]


        model_path, index_path = (
            _find_model_files(model_name)
        )


        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )


        print(
            "Loading RVC:",
            model_name,
            device
        )


        rvc = RVCInference(
            device=device
        )

        # Real-time-friendly f0 estimator.
        # "harvest" (the library default) is CPU-bound and far too slow
        # for the hop budget. rmvpe is fast and its weights are already
        # auto-downloaded by RVCInference.__init__.
        rvc.set_params(
            f0method="rmvpe"
        )


        rvc.load_model(
            model_path
        )


        rvc.models[
            rvc.current_model
        ]["index"] = index_path


        # --------------------------------------------------
        # Load HuBERT explicitly.
        #
        # rvc_python only loads hubert_model lazily, inside
        # VC.vc_single(). Since we call pipeline.pipeline()
        # directly (bypassing vc_single), that lazy load never
        # happens and vc.hubert_model stays None -> AttributeError
        # deep inside Pipeline.vc() the first time a frame arrives.
        # --------------------------------------------------
        rvc.vc.hubert_model = load_hubert(
            rvc.config,
            rvc.lib_dir
        )


        _MODEL_CACHE[model_name] = rvc


        print(
            "RVC loaded"
        )


        return rvc



# --------------------------------------------------
# RVC Stream
# --------------------------------------------------

# How much real (non-reflected) left-context to carry from one
# chunk into the next, measured at the pipeline's internal 16kHz
# rate. Prepending genuine prior audio instead of relying on
# Pipeline's reflect-padding gives HuBERT/RMVPE real continuity
# across chunk boundaries, instead of restarting "cold" on
# mirrored copies of the current chunk every 500ms.
CONTEXT_MS = 300
CONTEXT_SAMPLES_16K = int(16000 * CONTEXT_MS / 1000)

# Below this peak amplitude (on the raw 48kHz input), treat the
# chunk as silence/noise-floor and skip RVC entirely rather than
# let HuBERT/RMVPE hallucinate voiced content out of noise.
SILENCE_THRESHOLD = 0.03


class RVCStream:


    def __init__(
        self,
        model_name
    ):

        self.model = _load_model(
            model_name
        )

        # Real 16kHz audio carried over from the end of the
        # previous chunk, used as left-context for the next call
        # instead of reflect-padding. Starts as silence.
        self._context_16k = np.zeros(
            CONTEXT_SAMPLES_16K,
            dtype=np.float32
        )


    def process(
        self,
        audio: np.ndarray,
        input_sample_rate=48000,
        output_sample_rate=48000
    ):

        """
        Convert one audio window.

        Input:
            float32 mono waveform at input_sample_rate

        Output:
            float32 mono waveform at output_sample_rate, range [-1, 1]
        """

        audio = np.asarray(
            audio,
            dtype=np.float32
        )


        if audio.ndim != 1:

            raise ValueError(
                "Expected mono audio"
            )


        # --------------------------------------------------
        # The pipeline's HuBERT front-end, F0 estimator and
        # internal filters are hardwired to 16kHz
        # (see Pipeline.__init__: self.sr = 16000). Feeding it
        # 48kHz audio directly does not error, it just produces
        # garbage - the model never "expected" this because it
        # is silently treating your 48kHz samples as if they
        # were 16kHz.
        # --------------------------------------------------
        if input_sample_rate != 16000:

            audio_16k_new = librosa.resample(
                audio,
                orig_sr=input_sample_rate,
                target_sr=16000
            )

        else:

            audio_16k_new = audio


        # --------------------------------------------------
        # Silence gate. Skip RVC entirely on near-silent input -
        # still update the context buffer with this (quiet) real
        # audio so that when real speech resumes, the next call's
        # left-context is recent silence, not stale speech.
        # --------------------------------------------------
        is_silent = (
            np.abs(audio).max() < SILENCE_THRESHOLD
        )

        if is_silent:

            if len(audio_16k_new) >= CONTEXT_SAMPLES_16K:

                self._context_16k = audio_16k_new[-CONTEXT_SAMPLES_16K:]

            else:

                self._context_16k = np.concatenate([
                    self._context_16k[
                        len(audio_16k_new):
                    ],
                    audio_16k_new
                ])

            return np.zeros(
                int(len(audio) * output_sample_rate / input_sample_rate),
                dtype=np.float32
            )


        # --------------------------------------------------
        # Prepend real left-context instead of letting Pipeline
        # reflect-pad the chunk against itself.
        # --------------------------------------------------
        audio_16k = np.concatenate([
            self._context_16k,
            audio_16k_new
        ])

        context_len = len(self._context_16k)
        total_len = len(audio_16k)

        # Update context for the next call - tail of the NEW audio
        # only (not the context we just prepended), so it doesn't
        # grow.
        if len(audio_16k_new) >= CONTEXT_SAMPLES_16K:

            self._context_16k = audio_16k_new[-CONTEXT_SAMPLES_16K:]

        else:

            self._context_16k = np.concatenate([
                self._context_16k[
                    len(audio_16k_new):
                ],
                audio_16k_new
            ])


        # Mirror the normalization vc_single() does before
        # handing audio to the pipeline, to avoid clipping
        # into HuBERT.
        peak = np.abs(audio_16k).max()

        if peak > 0.95:

            audio_16k = audio_16k / (peak / 0.95)


        times = [
            0,
            0,
            0
        ]


        model = self.model


        current = (
            model.models[
                model.current_model
            ]
        )


        with torch.no_grad():

            output_int16 = model.vc.pipeline.pipeline(

                model.vc.hubert_model,

                model.vc.net_g,

                0,                      # speaker id

                audio_16k,

                None,                   # input path

                times,

                0,                      # pitch shift

                model.f0method,

                current.get(
                    "index",
                    None
                ),

                model.index_rate,

                model.vc.if_f0,

                model.filter_radius,

                model.vc.tgt_sr,

                model.resample_sr,

                model.rms_mix_rate,

                model.vc.version,

                model.protect,

                None                    # f0_file

            )


        # --------------------------------------------------
        # pipeline.pipeline() always returns int16 PCM, scaled
        # to the full 16-bit range - never a normalized float32
        # waveform. Casting straight to float32 with no /32768
        # leaves values around +/-32000 instead of +/-1, which
        # is why playback was blown-out noise / silence-via-clip.
        # --------------------------------------------------
        output_float = (
            output_int16.astype(np.float32) / 32768.0
        )


        # --------------------------------------------------
        # Trim off the portion of the output corresponding to the
        # prepended context, proportionally to how much of the
        # 16kHz input it represented. The pipeline's frame-based
        # internals (HuBERT @ 50Hz, net_g hop) mean this ratio is
        # an approximation, not an exact sample-accurate cut, but
        # it's close enough to avoid re-emitting the context audio
        # on every chunk.
        # --------------------------------------------------
        trim_samples = int(
            len(output_float) * context_len / total_len
        )

        output_float = output_float[trim_samples:]


        # resample_sr == 0 means "don't force a resample", in
        # which case the pipeline's real output rate is tgt_sr
        # (the model's training sample rate). Only when
        # resample_sr is set (and >=16000) does the pipeline
        # resample internally to that rate instead.
        if model.resample_sr >= 16000:

            tgt_sr = model.resample_sr

        else:

            tgt_sr = model.vc.tgt_sr


        if tgt_sr != output_sample_rate:

            output_float = librosa.resample(
                output_float,
                orig_sr=tgt_sr,
                target_sr=output_sample_rate
            )


        return output_float



# --------------------------------------------------
# ZeroMQ worker
# --------------------------------------------------

def run_worker():

    SAMPLE_RATE = 48000


    rvc = RVCStream(
        "shylily"
    )


    context = zmq.Context()


    receiver = context.socket(
        zmq.PULL
    )

    receiver.connect(
        "tcp://127.0.0.1:5555"
    )


    sender = context.socket(
        zmq.PUSH
    )

    sender.bind(
        "tcp://127.0.0.1:5556"
    )


    print(
        "RVC worker ready"
    )


    while True:

        packet = receiver.recv()

        audio = np.frombuffer(
            packet,
            dtype=np.float32
        )

        print(
            "Input:",
            audio.shape,
            "peak:",
            np.abs(audio).max(),
            flush=True
        )

        start = time.time()

        converted = rvc.process(
            audio,
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE
        )

        elapsed = time.time() - start

        print(
            f"[rvc] process() took {elapsed * 1000:.0f}ms",
            flush=True
        )

        sender.send(
            converted.astype(
                np.float32
            ).tobytes()
        )



# --------------------------------------------------
# Entry point
#
# The one-shot warmup test (silence in -> process() -> print
# shape/dtype/peak) has moved to test_rvc.py, since main.py
# runs `python -m tools.rvc` expecting this to start the actual
# PULL/PUSH worker loop and block, not run one test and exit.
# --------------------------------------------------

if __name__ == "__main__":

    run_worker()