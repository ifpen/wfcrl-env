import argparse
import platform
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Install simulators")
parser.add_argument(
    "simulator", type=str, help="Name of the simulator. Accepted: `fastfarm`"
)


def install_simulator(simulator=None):
    if simulator is None:
        # if not provided, retrieve from sys.argv
        if len(sys.argv) < 2:
            raise ValueError("No simulator specified.")
        simulator = sys.argv[1]
    if simulator == "floris":
        raise Warning("Similator FLORIS already installed with WFCRL.")
    elif simulator != "fastfarm":
        raise ValueError(f"Unknown simulator name {simulator}")
    syst = platform.system().lower()
    if syst == "windows":
        raise Warning(
            "Automatic installation not yet supported on Windows."
            "Please download pre-compiled FAST.Farm binaries from "
            "https://github.com/OpenFAST/openfast/releases/tag/v3.5.1"
        )
    else:
        # Run bash install file, will:
        # 1. Install OPENFAST (assume cmake is here already)
        # 2. Make DLL files with cmake => install in wfcrl/simulators/fastfarm/servo_dll
        path_to_install_script = Path(__file__).parent / "make_ff.sh"
        print(f"Will run install script {path_to_install_script}")
        subprocess.run(["bash", str(path_to_install_script)])


if __name__ == "__main__":
    args = parser.parse_args()
    install_simulator(args.simulator)
