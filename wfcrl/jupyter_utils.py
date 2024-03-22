import json
import os
import subprocess
import sys

kernel_json_params = {
    "argv": [
        "mpiexec",
        "-n",
        "1",
        "path_to_python.exe",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ],
    "display_name": "WFCRL Interface",
    "language": "python",
    "metadata": {"debugger": True},
}
kernel_json_params["argv"][3] = sys.executable


def create_ipykernel():
    os.makedirs("wfcrl-kernel", exist_ok=True)
    with open("wfcrl-kernel/kernel.json", "w") as fp:
        json.dump(kernel_json_params, fp)
    subprocess.run(("jupyter kernelspec install --user wfcrl-kernel").split(" "))
