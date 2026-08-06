import sounddevice as sd
import numpy as np
from collections import deque
from queue import Queue
import threading


SAMPLE_RATE = 48000

WINDOW_MS = 320
STEP_MS = 50

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_MS / 1000)


class AudioInput:
    """
    Real-time microphone input.

    Produces overlapping audio windows:
        - Window size: 320ms
        - Step size: 50ms

    Output:
        numpy.ndarray(float32)
        Shape: (WINDOW_SAMPLES,)
    """

    def __init__(self):
        self.queue = Queue(maxsize=10)

        self.buffer = deque(
            maxlen=WINDOW_SAMPLES
        )

        self.running = False
        self.stream = None


    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        # Convert mono microphone input
        samples = indata[:, 0].copy()

        self.buffer.extend(samples)

        # Only emit when enough samples exist
        if len(self.buffer) == WINDOW_SAMPLES:

            window = np.asarray(
                self.buffer,
                dtype=np.float32
            )

            # Prevent queue buildup
            if not self.queue.full():
                self.queue.put(window)


    def start(self):
        """
        Start microphone capture.
        """

        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=STEP_SAMPLES,
            dtype="float32",
            callback=self._callback
        )

        self.stream.start()

        print(
            f"Input started "
            f"({WINDOW_MS}ms window / {STEP_MS}ms step)"
        )


    def read(self):
        """
        Get the newest audio window.

        Blocks until audio is available.
        """

        return self.queue.get()


    def stop(self):
        """
        Stop microphone capture.
        """

        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


if __name__ == "__main__":
    """
    Simple input test.
    """

    import time

    mic = AudioInput()
    mic.start()

    try:
        while True:
            audio = mic.read()

            print(
                "Received:",
                len(audio),
                "samples",
                f"({len(audio)/SAMPLE_RATE*1000:.1f}ms)"
            )

    except KeyboardInterrupt:
        mic.stop()
        print("Stopped")