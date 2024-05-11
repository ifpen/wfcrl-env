# WFCRL: Benchmark Reinforcement Learning for Wind Farm Control

## Farm environments

List all environments:

```
from wfcrl import environments as envs
envs.list_envs()
```


|  Root Name |  Description |
|---|---|
|  Ablaincourt |  Inspired by layout of the Ablaincourt farm in France, (Duc et al, 2019) |
|  Turb16_TCRWP |  Layout of the [Total Control Reference Wind Power Plant](https://farmconners.readthedocs.io/en/latest/provided_data_sets.html) (TC RWP) (the first 16 turbines) |
| Turb3_Row1  |  Custom case - Single row of 3 turbines |
| Turb6_Row2   |  Custom case - 2 rows of 6 turbine |
| Turb16_Row5   | Layout of the first 16 turbines in the the CL-Windcon project [as implemented in WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim/blob/master/layoutDefinitions/layoutSet_clwindcon_80turb.m)|
| Turb32_Row5   | Layout of the first 32 turbines in the the CL-Windcon project [as implemented in WFSim](https://github.com/TUDelft-DataDrivenControl/WFSim/blob/master/layoutDefinitions/layoutSet_clwindcon_80turb.m)  |
| TurbX_Row1 for X in [1, 12] | Procedurally generated single row layout with X turbines  |

All wind farms environments are implemented with both the `Gymnasium` and `PettingZoo` API, and can be run on both the `Floris` and the `FastFarm` wind farm simulators.


The root name of the environment is associated with a prefix and a suffix:
- A `Dec_` prefix is added before environment names to indicate an Agent Environment Cycle implementation supported by `PettingZoo`.
- A `Floris` or `Fastfarm` suffix is added after the name of the environment to indicate the name of the background simulator.

## Example

Creating a wind farm environment of the TC RWP layout with the Floris background on Gymnasium:

```
from wfcrl import environments as envs
env = envs.make("Ablaincourt_Floris")
```

To use the Fastfarm background, make sure to have installed the Fastfarm and MPI dependencies installed as indicated in the `README.md`.

## Notebook

More detailed examples can be found in the `demo.ipynb` notebook.

To run the `demo.ipynb` notebook, you will first need to install the WFCRL kernel:

- Install `jupyter notebook` and `seaborn`:

```
pip install notebook seaborn
```

- Install the jupyter kernel

```
from wfcrl import jupyter_utils
jupyter_utils.create_ipykernel()
```
