# WFCRL: Interfacing and Benchmark Reinforcement Learning for Wind Farm Control

## Using the FastFarm interface

### Installation

This assumes that you already have compiled FAST.Farm binaries. If not you can download them [here](https://github.com/OpenFAST/openfast/releases/tag/v3.5.1). This interface has been tested with FAST.Farm 3.5.1.

1. Download **BOTH** Windows MPI setup (.exe) and MPI SDK (.msi) and install them from (https://www.microsoft.com/en-us/download/details.aspx?id=100593)
You can check your installation by enBtering : `set MSMPI` from `C:\Windows\System32` in the command prompt. You should obtain the following:

```
MSMPI_BENCHMARKS=C:\Program Files\Microsoft MPI\Benchmarks\
MSMPI_BIN=C:\Program Files\Microsoft MPI\Bin\
MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include\
MSMPI_LIB32=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86\
MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\
```

2. In the virtual environment of your choice:
```
pip install -r requirements.txt
```

## Usage

### Interfacing with FAST.Farm

A simple tutorial to start a simulation with the FastFarm interface is available in the notebook `interface.ipynb` notebook. To properly launch the notebook, see the intructions below in *Running Examples Notebook*.

**Creating an interface from a WFCRL case:**

```
from wfcrl.environments import data_cases as cases
from wfcrl.interface import FastFarmInterface

config = cases.fastfarm_6t
interface = FastFarmInterface(config)
```

By default, your FAST.Farm executable is assumed to be located in `simulators/fastfarm/bin/FAST.Farm_x64_OMP_2023.exe`. If not, you can also pass it to the interface:

```
interface = FastFarmInterface(config, fast_farm_executable=path_to_exe)
```


**Creating an interface from existing configuration files:**
Alternatively, if you already have your simulation fils ready, you can just point towards the `.fstf` file:
```
ff_interface = FastFarmInterface(fstf_file=path_to_fstf)
```


At every iteration, the FAST.Farm interace retrieves 12 measures per turbine:
- 2 wind measurements: wind velocity and direction at the entrance of the farm
- The current output power of the turbine
- The yaw of the turbine
- The pitch of the turbine
- The torque of the turbine
- 6 measures of blade loads

### Using the RL environments
An example of a test case using the multi-agent environment with a simple step policy is given in `example.py`. It can be launched with

```
mpiexec -n 1 python example.py
```

A more detailed introduction is available in the `demo.ipynb` notebook. To properly launch the notebook, see the intructions below in *Running Examples Notebook*.

## Running Examples Notebook

To run the `interface.ipynb` and `demo.ipynb` examples, you will first need to install the WFCRL kernel:

```
from wfcrl import jupyter_utils
jupyter_utils.create_ipykernel()
```
