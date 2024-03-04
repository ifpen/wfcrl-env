# WFCRL: Interfacing and Benchmark Reinforcement Learning for Wind Farm Control

## Using the FastFarm interface

### Installation

This assumes that you already have compiled FAST.Farm binaries. If not you can download them [here](https://github.com/OpenFAST/openfast/releases/tag/v3.5.1). This interface has been tested with FAST.Farm 3.5.1.

1. Download **BOTH** Windows MPI setup (.exe) and MPI SDK (.msi) and install them from (https://www.microsoft.com/en-us/download/details.aspx?id=100593)
You can check your installation by entering : `set MSMPI` from `C:\Windows\System32` in the command prompt. You should obtain the following:

```
MSMPI_BENCHMARKS=C:\Program Files\Microsoft MPI\Benchmarks\
MSMPI_BIN=C:\Program Files\Microsoft MPI\Bin\
MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include\
MSMPI_LIB32=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86\
MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\
```

2. Create a `.env` file at the root of the folder to store the path towards the FAST.Farm executable. For example:

```
FAST_FARM_EXECUTABLE = simulators/fastfarm/bin/FAST.Farm_x64_OMP_2023.exe
```

### Usage

An example can be seen in the `interface.ipynb` notebook. At every iteration, the FAST.Farm interace retrieves 12 measures per turbine:
- 2 wind measurements: wind velocity and direction at the entrance of the farm
- The current output power of the turbine
- The yaw of the turbine
- The pitch of the turbine
- The torque of the turbine
- 6 measures of blade loads

## Running Examples Notebook

To run the `interface.ipynb` and `demo.ipynb` examples, you will need to launch the notebooks with `MPI`:

Run the following command to find the root folder of your kernel specifications.
```
jupyter kernelspec list
```

Open the `kernel.json` file, and in `argv`, add at the beginning of the list, add the following arguments `"mpiexec", "-n", "1",`. Your complete argument list should look like this:

```
"argv": [
  "mpiexec", "-n", "1",
  path_to_python.exe,
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
``
