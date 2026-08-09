"""
RVC inference worker — runs inside the "rvc" conda env.

Loads the RVC model once, then services one-shot conversion requests
over stdin/stdout so the main push-to-talk process (running in a
different conda env, for sounddevice/keyboard) can hand it work
without needing rvc-python installed in its own environment.

Protocol (line-based, one request per line):
    stdin:  <path to input wav>
    stdout: OK <path to output wav>      on success
            ERR <message>                on failure

Run directly as a subprocess — do not import.
"""

import argparse
import contextlib
import os
import sys
import tempfile

# Keep a handle to the *real* stdout before anything below can touch it.
# This is the only channel we use for the OK/ERR protocol.
_real_stdout = sys.stdout


def _send(line: str):
    _real_stdout.write(line + "\n")
    _real_stdout.flush()


# PyTorch 2.6 flipped torch.load's default to weights_only=True, which
# breaks fairseq's hubert checkpoint loading (it needs to unpickle a
# fairseq.data.dictionary.Dictionary). This model file is your own
# local checkpoint, so it's safe to force weights_only=False.
import torch  # noqa: E402

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from rvc_python.infer import RVCInference  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--f0method", default="rmvpe")
    parser.add_argument("--f0up_key", type=int, default=0, help="pitch shift in semitones, +up/-down")
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--rms_mix_rate", type=float, default=0.25)
    args = parser.parse_args()

    print(f"[worker] loading model {args.model}...", file=sys.stderr, flush=True)
    # rvc_python/fairseq print stray diagnostic lines straight to stdout
    # (e.g. "is_half:True, device:cuda:0"). stdout is our protocol
    # channel, so redirect any of their incidental prints to stderr for
    # the duration of these calls.
    with contextlib.redirect_stdout(sys.stderr):
        rvc = RVCInference(device=args.device)
        rvc.load_model(args.model, index_path=args.index)
        rvc.set_params(
            f0method=args.f0method,
            f0up_key=args.f0up_key,
            index_rate=args.index_rate,
            protect=args.protect,
            filter_radius=args.filter_radius,
            rms_mix_rate=args.rms_mix_rate,
        )
    print("[worker] ready", file=sys.stderr, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="rvc_worker_")

    # Each line on stdin is a path to a wav file to convert.
    for line in sys.stdin:
        in_path = line.strip()
        if not in_path:
            continue
        try:
            out_path = os.path.join(tmp_dir, f"out_{os.path.basename(in_path)}")
            with contextlib.redirect_stdout(sys.stderr):
                rvc.infer_file(in_path, out_path)
            _send(f"OK {out_path}")
        except Exception as e:
            _send(f"ERR {e}")


if __name__ == "__main__":
    main()