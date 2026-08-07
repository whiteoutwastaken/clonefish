import sounddevice as sd
import numpy as np
from collections import deque
import zmq

SAMPLE_RATE = 48000

INPUT_DEVICE = 44  # or whichever index matches your actual mic

# Raised from 320/50 - the RVC worker's steady-state process()
# cost measured ~330-350ms per call (see test_rvc.py), so a 50ms
# hop meant the worker fell further behind with every chunk.
# WINDOW_MS is kept equal to STEP_MS (no overlap) since output.py
# currently plays whatever it receives untrimmed - overlapping
# windows would mean hearing duplicated audio at each boundary
# until a crossfade/trim step exists on the output side.
WINDOW_MS = 500
STEP_MS = 500

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_MS / 1000)


class AudioInput:

    def __init__(self):

        self.buffer = deque(
            maxlen=WINDOW_SAMPLES
        )

        # -----------------------------------
        # ZeroMQ
        # -----------------------------------

        self.context = zmq.Context()

        self.socket = self.context.socket(
            zmq.PUSH
        )

        # Sends audio to the RVC process
        self.socket.bind(
            "tcp://127.0.0.1:5555"
        )

        self.running = False
        self.stream = None


    def _callback(self, indata, frames, time_info, status):

        if status:
            print(status)

        samples = indata[:, 0].copy()

        self.buffer.extend(samples)

        if len(self.buffer) != WINDOW_SAMPLES:
            return

        window = np.asarray(
            self.buffer,
            dtype=np.float32
        )

        # Send raw bytes
        self.socket.send(
            window.tobytes()
        )


    def start(self):

        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=STEP_SAMPLES,
            dtype="float32",
            device=INPUT_DEVICE,
            callback=self._callback
        )

        self.stream.start()

        print(
            f"Input started ({WINDOW_MS} ms / {STEP_MS} ms)"
        )


    def stop(self):

        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.socket.close()
        self.context.term()


if __name__ == "__main__":

    import time

    mic = AudioInput()

    mic.start()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        mic.stop()