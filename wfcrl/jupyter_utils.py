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
    os.makedirs("kernel-wfcrl", exist_ok=True)
    with open("kernel-wfcrl/kernel.json", "w") as fp:
        json.dump(kernel_json_params, fp)
    subprocess.run(("jupyter kernelspec install --user kernel-wfcrl").split(" "))
