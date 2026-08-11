"""
Push-to-talk RVC voice changer — controller process.

Run this in your "sound-input" conda env (needs sounddevice, soundfile,
numpy, keyboard). It handles the global hotkey, recording, and
playback. Actual RVC conversion is delegated to rvc_worker.py, which
this script launches as a subprocess using the "rvc" conda env's
python interpreter, so the two environments never need to share
dependencies. They talk over the worker's stdin/stdout — one request
per hotkey release, no streaming, no sockets.

Hold HOTKEY anywhere in Windows to record. On release, the whole clip
is sent to the worker for one conversion pass, then played out to your
VB-Audio virtual mic.
"""

import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard

# ----------------------------------------------------------------------
# CONFIG - edit these
# ----------------------------------------------------------------------
# Full path to python.exe inside your "rvc" conda env. Find it with:
#   conda run -n rvc python -c "import sys; print(sys.executable)"
RVC_PYTHON = r"C:\Users\whiteout\miniconda3\envs\rvc\python.exe"
RVC_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_worker.py")

MODEL_PATH = r"C:\Users\whiteout\Documents\Projects\clonefish\models\egirl/egirl.pth"
INDEX_PATH = r"C:\Users\whiteout\Documents\Projects\clonefish\models\egirl/egirl.index"   # or None
RVC_DEVICE = "cuda:0"
F0_METHOD = "rmvpe"

# Voice quality / pitch tuning — see rvc_worker.py for what each does.
PITCH_SHIFT = 15        # semitones, + = higher pitch, - = lower. e.g. 4 for a noticeably higher voice
INDEX_RATE = 0.5        # 0-1, how much of the target voice's real timbre to pull from the index
PROTECT = 0.15          # 0-0.5, protects consonants/breath from artifacting
FILTER_RADIUS = 3       # >=3 enables median filtering on pitch, smooths jitter
RMS_MIX_RATE = 0.25     # 0-1, loudness envelope blend; lower = more of your natural dynamics

HOTKEY = "`"
SAMPLE_RATE = 44100
MIC_SAMPLE_RATE = 48000
CHANNELS = 1

INPUT_DEVICE = None      # your mic, sounddevice index. None = OS default.
OUTPUT_DEVICE = None     #  "headset", sounddevice index.
MIC_DEVICE = 39          # VB CABLE , sounddevice index. None = OS default.
PRINT_DEVICES_ONLY = False
# ----------------------------------------------------------------------

BEEP_FREQ = 680          # Hz, tone pitch
BEEP_DURATION = 0.42     # seconds, how long the beep plays
BEEP_VOLUME = 0.1        # 0-1
PRE_PLAY_DELAY = 0.3     # seconds of silence after the beep, before the converted audio plays

from scipy.signal import resample_poly
from math import gcd

def _resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)

class RVCWorkerClient:
    """Launches and talks to rvc_worker.py running in the rvc conda env."""

    def __init__(self):
        cmd = [
            RVC_PYTHON, RVC_WORKER_SCRIPT,
            "--model", MODEL_PATH,
            "--device", RVC_DEVICE,
            "--f0method", F0_METHOD,
            "--f0up_key", str(PITCH_SHIFT),
            "--index_rate", str(INDEX_RATE),
            "--protect", str(PROTECT),
            "--filter_radius", str(FILTER_RADIUS),
            "--rms_mix_rate", str(RMS_MIX_RATE),
        ]
        if INDEX_PATH:
            cmd += ["--index", INDEX_PATH]

        # stderr is left un-piped so the worker's "[worker] loading
        # model..." / "[worker] ready" logs just print to this console.
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

    def convert(self, wav_path: str) -> str:
        with self._lock:
            self.proc.stdin.write(wav_path + "\n")
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline().strip()
        if reply.startswith("OK "):
            return reply[3:]
        raise RuntimeError(f"rvc worker error: {reply}")

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()


class PushToTalk:
    def __init__(self, worker: RVCWorkerClient):
        self.worker = worker
        self._recording = False
        self._chunks = []
        self._lock = threading.Lock()
        # Opened once and left running for the program's lifetime —
        # opening a stream per press/release was the source of the
        # startup latency you were hearing. The _recording flag alone
        # now controls whether incoming audio is kept or dropped.
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=INPUT_DEVICE,
            callback=self._callback,
        )
        self._stream.start()
    def _make_beep(self):
        t = np.linspace(0, BEEP_DURATION, int(SAMPLE_RATE * BEEP_DURATION), endpoint=False)
        tone = BEEP_VOLUME * np.sin(2 * np.pi * BEEP_FREQ * t)
        # short fade in/out so it doesn't click
        fade = min(200, len(tone) // 4)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone.astype(np.float32)
    
    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        with self._lock:
            if self._recording:
                self._chunks.append(indata.copy())

    def start_recording(self):
        with self._lock:
            if self._recording:
                return  # ignore key-repeat "down" events while already held
            self._chunks = []
            self._recording = True
        print("[recording...]")

    def stop_recording_and_process(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            chunks, self._chunks = self._chunks, []

        if not chunks:
            print("[nothing recorded]")
            return

        audio = np.concatenate(chunks, axis=0)
        print(f"[recorded {len(audio) / SAMPLE_RATE:.2f}s, converting...]")
        threading.Thread(target=self._convert_and_play, args=(audio,), daemon=True).start()

    def close(self):
        self._stream.stop()
        self._stream.close()

    def _convert_and_play(self, audio):
        t0 = time.time()
        tmp_dir = tempfile.gettempdir()
        in_path = os.path.join(tmp_dir, f"ptt_in_{int(t0 * 1000)}.wav")
        sf.write(in_path, audio, SAMPLE_RATE)

        try:
            out_path = self.worker.convert(in_path)
        except RuntimeError as e:
            print(f"[conversion failed: {e}]")
            return

        out_audio, out_sr = sf.read(out_path, dtype="float32")
        play_audio = _resample(out_audio, out_sr, MIC_SAMPLE_RATE)
        print(f"[converted in {time.time() - t0:.2f}s, playing]")
        sd.play(self._make_beep(), samplerate=SAMPLE_RATE, device=OUTPUT_DEVICE)
        sd.wait()
        time.sleep(PRE_PLAY_DELAY)
        print(f"[converted in {time.time() - t0:.2f}s, out_sr={out_sr}, playing]")
        sd.play(play_audio, samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE)
        sd.wait()
        print("[done]")


def main():
    if PRINT_DEVICES_ONLY:
        print(sd.query_devices())
        return

    print("[main] starting rvc worker subprocess (rvc conda env)...")
    worker = RVCWorkerClient()

    ptt = PushToTalk(worker)
    keyboard.on_press_key(HOTKEY, lambda e: ptt.start_recording())
    keyboard.on_release_key(HOTKEY, lambda e: ptt.stop_recording_and_process())

    print(f"Listening for '{HOTKEY}' globally. Hold to talk, release to convert. Ctrl+C to quit.")
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        ptt.close()
        worker.close()


if __name__ == "__main__":
    main()