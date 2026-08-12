import sounddevice as sd
import numpy as np
import zmq
import queue
from . import config

SAMPLE_RATE = config.SAMPLE_RATE

OUTPUT_DEVICE = config.OUTPUT_DEVICE


class AudioOutput:

    def __init__(self):

        self.audio_queue = queue.Queue()
        self._current_chunk = None
        self._chunk_pos = 0

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=config.get_input_channels(OUTPUT_DEVICE),
            dtype="float32",
            device=OUTPUT_DEVICE,
            latency="low",
            callback=self._callback
        )

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect("tcp://127.0.0.1:5556")

        self.running = False


    def _callback(self, outdata, frames, time_info, status):

        if status:
            print(status, flush=True)

        filled = 0

        while filled < frames:

            if self._current_chunk is None or self._chunk_pos >= len(self._current_chunk):

                try:
                    self._current_chunk = self.audio_queue.get_nowait()
                    self._chunk_pos = 0
                except queue.Empty:
                    outdata[filled:, 0] = 0
                    return

            remaining_in_chunk = len(self._current_chunk) - self._chunk_pos
            to_copy = min(frames - filled, remaining_in_chunk)

            outdata[filled:filled + to_copy, 0] = (
                self._current_chunk[self._chunk_pos:self._chunk_pos + to_copy]
            )

            self._chunk_pos += to_copy
            filled += to_copy


    def start(self):

        self.stream.start()
        print("Virtual microphone started", flush=True)


    def run(self):

        self.running = True
        print("Waiting for converted audio...", flush=True)

        while self.running:

            packet = self.socket.recv()
            audio = np.frombuffer(packet, dtype=np.float32)
            print(f"[output] queued {audio.shape[0]} samples, peak={np.abs(audio).max():.4f}", flush=True)
            self.audio_queue.put(audio)


    def stop(self):

        self.running = False
        self.stream.stop()
        self.stream.close()
        self.socket.close()
        self.context.term()



if __name__ == "__main__":

    out = AudioOutput()

    out.start()

    try:

        out.run()

    except KeyboardInterrupt:

        print(
            "Stopping..."
        )

        out.stop()