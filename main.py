import subprocess
import time
import sys


sound_input_python = (
    r"C:\Users\whiteout\miniconda3\envs\sound-input\python.exe"
)

rvc_python = (
    r"C:\Users\whiteout\miniconda3\envs\rvc\python.exe"
)

input_process = subprocess.Popen(
    [
        sound_input_python,
        "-m",
        "tools.input"
    ]
)


time.sleep(1)


rvc_process = subprocess.Popen(
    [
        rvc_python,
        "-m",
        "tools.rvc"
    ]
)


try:
    input_process.wait()
    rvc_process.wait()

except KeyboardInterrupt:
    input_process.terminate()
    rvc_process.terminate()