from tools.input import AudioInput
import numpy as np

mic = AudioInput()
mic.start()

try:
    while True:
        audio = mic.read()

        print(
            "RMS:",
            np.sqrt(np.mean(audio ** 2))
        )

except KeyboardInterrupt:
    mic.stop()