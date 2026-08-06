"""
Real-time RVC wrapper.

Input:
    numpy float32 audio arrays

Output:
    numpy float32 converted audio arrays
"""


from pathlib import Path
import threading

import numpy as np
import torch


# --------------------------------------------------
# PyTorch 2.6+ Compatibility Fix
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


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR
    / "voice"
    / "models"
)


# --------------------------------------------------
# Cache
# --------------------------------------------------

_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()



def _find_model_files(model_name):

    folder = MODEL_DIR / model_name

    if not folder.exists():
        raise FileNotFoundError(
            folder
        )


    pth = list(
        folder.glob("*.pth")
    )

    index = list(
        folder.glob("*.index")
    )


    if len(pth) != 1:
        raise RuntimeError(
            "Expected exactly one .pth file"
        )


    if len(index) != 1:
        raise RuntimeError(
            "Expected exactly one .index file"
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


        rvc.load_model(
            model_path
        )


        rvc.models[
            rvc.current_model
        ]["index"] = index_path


        _MODEL_CACHE[model_name] = rvc


        print("RVC loaded")


        return rvc



# --------------------------------------------------
# Public Interface
# --------------------------------------------------

class RVCStream:


    def __init__(
        self,
        model_name
    ):

        self.model = _load_model(
            model_name
        )


    def process(
        self,
        audio: np.ndarray,
        sample_rate=48000
    ):
        """
        Convert one audio window.

        audio:
            float32 mono waveform
        """


        audio = np.asarray(
            audio,
            dtype=np.float32
        )


        with torch.no_grad():

            if hasattr(
                self.model,
                "infer"
            ):

                output = self.model.infer(
                    audio,
                    sample_rate
                )


            elif hasattr(
                self.model,
                "infer_audio"
            ):

                output = self.model.infer_audio(
                    audio,
                    sample_rate
                )


            else:

                raise RuntimeError(
                    "rvc-python does not expose "
                    "array inference. Need internal API."
                )


        return np.asarray(
            output,
            dtype=np.float32
        )