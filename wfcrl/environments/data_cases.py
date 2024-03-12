from dataclasses import dataclass
from typing import Callable, List, Union


def repr(self):
    repr = f"Wind farm simulation on {self.simulator}: "
    repr += f"{self.n_turbines} turbines\n"
    for arg, val in self.interface_kwargs.items():
        if arg[:2] != "__":
            if isinstance(val, dict):
                repr += f"{arg}: \n"
                for name, param in val.items():
                    repr += f"\t{name}: {param}\n"
            else:
                repr += f"{arg}: {val}\n"
    return repr


@dataclass
class DefaultControl:
    yaw = (-20, 20, 5)
    pitch = (0, 45, 1)
    torque = (-2e4, 2e4, 1e3)


@dataclass
class FarmCase:
    n_turbines: int
    simulator: str

    xcoords: Union[List, Callable]
    ycoords: Union[List, Callable]

    dt: int
    buffer_window: int
    t_init: int

    def __repr__(self):
        return repr(self)


class FastFarmCase(FarmCase):
    @property
    def interface_kwargs(self):
        return {
            "measurement_window": int(self.buffer_window / self.dt),
            "simul_params": self.simul_params,
        }

    @property
    def measurement_window(self):
        return self.interface_kwargs["measurement_window"]

    @property
    def simul_params(self):
        return {"xcoords": self.xcoords, "ycoords": self.ycoords, "dt": self.dt}


class FlorisCase(FarmCase):
    @property
    def interface_kwargs(self):
        return {"simul_params": self.simul_params}

    @property
    def simul_params(self):
        return {"xcoords": self.xcoords, "ycoords": self.ycoords, "direction": 270}


# 3 turbines row layouts
fastfarm_3t = FastFarmCase(
    simulator="FastFarm",
    n_turbines=6,
    xcoords=[0.0, 504.0, 1008.0],
    ycoords=[0.0, 0.0, 0.0],
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_3t = FlorisCase(
    simulator="Floris",
    n_turbines=3,
    xcoords=[400.0, 1000.0, 1600.0],
    ycoords=[400.0, 400.0, 400.0],
    dt=1,
    buffer_window=1,
    t_init=0,
)

# 2 x 3 layouts
fastfarm_6t = FastFarmCase(
    simulator="FastFarm",
    n_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[-252, -252, -252, 252, 252, 252],
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_6t = FlorisCase(
    simulator="Floris",
    n_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[400.0, 400.0, 400.0, 400.0, 400.0, 400.0],
    dt=1,
    buffer_window=1,
    t_init=0,
)


# TCRWPF layouts
#  Layout of the Total Control Reference Wind Power Plant (TC RWP) (first 16 turbines).
# https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
fastfarm_16TCRWP = FastFarmCase(
    simulator="FastFarm",
    n_turbines=16,
    xcoords=[(i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(16)],
    ycoords=[-300 + (i % 4) * 126 * 4 for i in range(16)],
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_16TCRWP = FlorisCase(
    simulator="Floris",
    n_turbines=16,
    xcoords=[(i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(16)],
    ycoords=[-300 + (i % 4) * 126 * 4 for i in range(16)],
    dt=1,
    buffer_window=1,
    t_init=0,
)

# Ablaincourt layouts
# from the layout of
# T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
# turbulence parameterization improves the jensen wake model and its
# implementation for power optimization of an operating wind farm"
fastfarm_ablaincourt = FastFarmCase(
    simulator="FastFarm",
    n_turbines=7,
    xcoords=[484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0],
    ycoords=[274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0],
    dt=3,
    buffer_window=600,
    t_init=700,
)
floris_ablaincourt = FlorisCase(
    simulator="Floris",
    n_turbines=7,
    xcoords=[484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0],
    ycoords=[274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0],
    dt=1,
    buffer_window=1,
    t_init=0,
)


class FarmRowFastfarm(FastFarmCase):
    """
    Base Layout.
    Simple farm with M aligned turbines.
    """

    dt = 3
    buffer_window = 600
    t_init = 700

    def get_xcoords(n_turbines):
        return [i * 504.0 for i in range(n_turbines)]

    def get_ycoords(n_turbines):
        return [0.0 for _ in range(n_turbines)]


class FarmRowFloris(FlorisCase):
    """
    Base Layout.
    Simple farm with M aligned turbines.
    """

    dt = 1
    buffer_window = 1
    t_init = 0

    def get_xcoords(n_turbines):
        return [400 + i * 600 for i in range(n_turbines)]

    def get_ycoords(n_turbines):
        return [400.0 for _ in range(n_turbines)]


named_cases_dictionary = {
    "Turb16_TCRWP_": [fastfarm_16TCRWP, floris_16TCRWP],
    "Turb3_Row1_": [fastfarm_3t, floris_3t],
    "Turb6_Row2_": [fastfarm_6t, floris_6t],
    "Ablaincourt_": [fastfarm_ablaincourt, floris_ablaincourt],
}
