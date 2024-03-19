from dataclasses import dataclass
from typing import Callable, List, Union


def repr(self):
    repr = f"Wind farm simulation on {self.simulator}: "
    repr += f"{self.num_turbines} turbines - {self.max_iter} timesteps\n"
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
    num_turbines: int

    xcoords: Union[List, Callable]
    ycoords: Union[List, Callable]

    dt: int
    buffer_window: int = 300
    t_init: int = 300
    max_iter: int = 100

    @property
    def interface_kwargs(self):
        return None

    def __repr__(self):
        return repr(self)

    def dict(self):
        return self.interface_kwargs


class FastFarmCase(FarmCase):
    simulator: str = "FastFarm"

    @property
    def interface_kwargs(self):
        params = {
            "max_iter": self.max_iter,
            "num_turbines": self.num_turbines,
        }
        params.update(self.simul_params)
        return params

    @property
    def avg_window(self):
        return int(self.buffer_window / self.dt)

    @property
    def simul_params(self):
        return {"xcoords": self.xcoords, "ycoords": self.ycoords, "dt": self.dt}


class FlorisCase(FarmCase):
    simulator: str = "Floris"

    @property
    def interface_kwargs(self):
        return self.simul_params

    @property
    def simul_params(self):
        return {
            "xcoords": self.xcoords,
            "ycoords": self.ycoords,
            "direction": 270,
            "speed": 8,
        }


# 3 turbines row layouts
fastfarm_3t = FastFarmCase(
    num_turbines=6,
    xcoords=[0.0, 504.0, 1008.0],
    ycoords=[0.0, 0.0, 0.0],
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_3t = FlorisCase(
    num_turbines=3,
    xcoords=[400.0, 1000.0, 1600.0],
    ycoords=[400.0, 400.0, 400.0],
    dt=1,
    buffer_window=1,
    t_init=0,
)

# 2 x 3 layouts
fastfarm_6t = FastFarmCase(
    num_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[-252, -252, -252, 252, 252, 252],
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_6t = FlorisCase(
    num_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[400.0, 400.0, 400.0, 400.0, 400.0, 400.0],
    dt=1,
    buffer_window=1,
    t_init=0,
)


# 16 turb layouts
# Used in Bizon Monroc, C., Bušić, A., Dubuc, D., & Zhu, J.
# "Actor critic agents for wind farm control", 2023
# fmt: off
xcoords = [
    891.5, 891.5, 891.5, 2088.6, 2089.1, 2088.2, 3285.7, 3285.3,
    3285.3, 3285.3, 3285.7, 4482.4, 4482.4, 4482.4, 4482.8, 4481.9,
]
ycoords = [
    5169.3, 3743.2, 2317.2, 5168.7, 3743.5, 2318.0, 6594.1, 5169.4,
    3743.4, 2317.3, 892.2, 6594.9, 5168.8, 3742.8, 2317.6, 892.1,
]
# fmt: on
fastfarm_16t = FastFarmCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_16t = FlorisCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=1,
    buffer_window=1,
    t_init=0,
)

# 32 turb layouts
# Used in Bizon Monroc, C., Bušić, A., Dubuc, D., & Zhu, J.
# "Actor critic agents for wind farm control", 2023
# fmt: off
xcoords = [
    891.5, 891.5, 891.5, 2088.6, 2089.1, 2088.2, 3285.7, 3285.3,
    3285.3, 3285.3, 3285.7, 4482.4, 4482.4, 4482.4, 4482.8, 4481.9,
    5679.5, 5679.9, 5679.1, 5679.0, 5679.5, 6876.2, 6876.2, 6876.1,
    6876.6, 6876.6, 8073.2, 8073.2, 8073.7, 8072.8, 8073.3, 9270.8,
]
ycoords = [
    5169.3, 3743.2, 2317.2, 5168.7, 3743.5, 2318.0, 6594.1, 5169.4,
    3743.4, 2317.3, 892.2, 6594.9, 5168.8, 3742.8, 2317.6, 892.1,
    6594.2, 5169.1, 3743.5, 2317.5, 892.3, 6595.0, 5169.0, 3742.9,
    2317.7, 891.7, 6594.4, 5168.3, 3743.2, 2317.6, 892.4, 6594.6,
]
# fmt: on
fastfarm_32t = FastFarmCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_32t = FlorisCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=1,
    buffer_window=1,
    t_init=0,
)

# TCRWPF layouts
#  Layout of the Total Control Reference Wind Power Plant (TC RWP) (first 16 turbines).
# https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
xcoords = ([(i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(16)],)
ycoords = ([-300 + (i % 4) * 126 * 4 for i in range(16)],)
fastfarm_16TCRWP = FastFarmCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=600,
    t_init=9,
)
floris_16TCRWP = FlorisCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
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
    num_turbines=7,
    xcoords=[484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0],
    ycoords=[274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0],
    dt=3,
    buffer_window=600,
    t_init=700,
)
floris_ablaincourt = FlorisCase(
    num_turbines=7,
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

    def get_xcoords(num_turbines):
        return [i * 504.0 for i in range(num_turbines)]

    def get_ycoords(num_turbines):
        return [0.0 for _ in range(num_turbines)]


class FarmRowFloris(FlorisCase):
    """
    Base Layout.
    Simple farm with M aligned turbines.
    """

    dt = 1
    buffer_window = 1
    t_init = 0

    def get_xcoords(num_turbines):
        return [400 + i * 600 for i in range(num_turbines)]

    def get_ycoords(num_turbines):
        return [400.0 for _ in range(num_turbines)]


named_cases_dictionary = {
    "Turb16_TCRWP_": [fastfarm_16TCRWP, floris_16TCRWP],
    "Turb3_Row1_": [fastfarm_3t, floris_3t],
    "Turb6_Row2_": [fastfarm_6t, floris_6t],
    "Turb16_Row5_": [fastfarm_16t, floris_16t],
    "Turb32_Row5_": [fastfarm_32t, floris_32t],
    "Ablaincourt_": [fastfarm_ablaincourt, floris_ablaincourt],
}
