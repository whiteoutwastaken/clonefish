import subprocess
import time
import os


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


sound_input_python = (
    r"C:\Users\whiteout\miniconda3\envs\sound-input\python.exe"
)

rvc_python = (
    r"C:\Users\whiteout\miniconda3\envs\rvc\python.exe"
)


def main():

    # Start microphone input
    input_process = subprocess.Popen(
        [
            sound_input_python,
            "-m",
            "tools.input"
        ],
        cwd=ROOT
    )


    time.sleep(1)


    # Start RVC
    rvc_process = subprocess.Popen(
        [
            rvc_python,
            "-m",
            "tools.rvc"
        ],
        cwd=ROOT
    )


    time.sleep(2)


    # Start virtual microphone output
    output_process = subprocess.Popen(
        [
            sound_input_python,
            "-m",
            "tools.output"
        ],
        cwd=ROOT
    )


    processes = [
        input_process,
        rvc_process,
        output_process
    ]


    try:

        while True:

            for process in processes:

                code = process.poll()

                if code is not None:

                    raise RuntimeError(
                        f"{process.args} exited early with code {code}"
                    )

            time.sleep(0.5)


    except KeyboardInterrupt:

        print("Stopping...")


    except RuntimeError as e:

        print(e)
        print("Stopping remaining processes...")


    finally:

        for process in processes:

            process.terminate()

        for process in processes:

            process.wait()



if __name__ == "__main__":

    main()