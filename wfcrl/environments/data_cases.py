from dataclasses import dataclass


def repr(self):
    repr = "Wind farm simulation configuration \n"
    repr = f"{self.n_turbines} turbines \n"
    for arg, val in self.interface_kwargs.items():
        if arg[:2] != "__":
            if isinstance(val, dict):
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
class Farm3Fastfarm:
    n_turbines = 3

    xcoords = [0.0, 504.0, 1008.0]
    ycoords = [0.0, 0.0, 0.0]

    dt = 3
    buffer_window = 600
    t_init = 9  # 100

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }


@dataclass()
class Farm6Fastfarm:
    """
    2 x 3 Layout.
    Used on FAST.Farm
    """

    n_turbines = 6

    xcoords = [0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0]
    ycoords = [-252, -252, -252, 252, 252, 252]

    dt = 3
    buffer_window = 600
    t_init = 9

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }
    interface_kwargs["log_file"] = "log.txt"

    def __repr__(self):
        return repr(self)


@dataclass
class Farm3Floris:
    """
    Base Layout.
    Simpe farm with 3 aligned turbines.
    """

    n_turbines = 3

    xcoords = [400.0, 1000.0, 1600.0]
    ycoords = [400.0, 400.0, 400.0]

    dt = 1
    buffer_window = 1
    t_init = 0

    interface_kwargs = {}
    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "direction": 270}
    interface_kwargs["simul_kwargs"] = simul_kwargs

    def __repr__(self):
        return repr(self)


@dataclass
class Farm6Floris:
    """
    2 x 3 Layout.
    Used on FAST.Farm
    """

    n_turbines = 6

    xcoords = [0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0]
    ycoords = [-252, -252, -252, 252, 252, 252]

    dt = 3
    buffer_window = 600
    t_init = 9

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }
    interface_kwargs["log_file"] = "log.txt"

    def __repr__(self):
        return repr(self)


@dataclass
class Farm16TCRWPFastfarm:
    """
    Layout of the Total Control Reference Wind Power Plant (TC RWP) (first 16 turbines).
    https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
    """

    n_turbines = 16

    xcoords = [
        (i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(n_turbines)
    ]
    ycoords = [-300 + (i % 4) * 126 * 4 for i in range(n_turbines)]

    dt = 3
    buffer_window = 600
    t_init = 100

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }

    def __repr__(self):
        return repr(self)


@dataclass
class Farm16TCRWPFloris:
    """
    Layout of the Total Control Reference Wind Power Plant (TC RWP) (first 16 turbines).
    https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
    """

    n_turbines = 16

    xcoords = [
        (i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(n_turbines)
    ]
    ycoords = [-300 + (i % 4) * 126 * 4 for i in range(n_turbines)]

    dt = 1
    buffer_window = 1
    t_init = 0

    interface_kwargs = {}
    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "direction": 270}
    interface_kwargs["simul_kwargs"] = simul_kwargs

    def __repr__(self):
        return repr(self)


@dataclass
class FarmAblaincourtFastfarm:
    """
    Layout of
    T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
    turbulence parameterization improves the jensen wake model and its
    implementation for power optimization of an operating wind farm"
    """

    n_turbines = 7

    xcoords = [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0]
    ycoords = [274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0]

    dt = 3
    buffer_window = 600
    t_init = 100

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }

    def __repr__(self):
        return repr(self)


@dataclass
class FarmAblaincourtFloris:
    """
    Layout of
    T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
    turbulence parameterization improves the jensen wake model and its
    implementation for power optimization of an operating wind farm"
    """

    n_turbines = 7

    xcoords = [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0]
    ycoords = [933.0, 910.0, 725.9, 636.3, 546.5, 463.7, 400.0]

    dt = 1
    buffer_window = 1
    t_init = 0

    interface_kwargs = {}
    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "direction": 270}
    interface_kwargs["simul_kwargs"] = simul_kwargs

    def __repr__(self):
        return repr(self)


@dataclass
class FarmRowFastfarm:
    """
    Base Layout.
    Simpe farm with M aligned turbines.
    """

    # Placeholders for parameters as a function of M
    n_turbines = None
    xcoords = None
    ycoords = None

    def get_xcoords(n_turbines):
        return [i * 504.0 for i in range(n_turbines)]

    def get_ycoords(n_turbines):
        return [0.0 for _ in range(n_turbines)]

    dt = 3
    buffer_window = 600
    t_init = 9  # 100

    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "dt": dt}
    interface_kwargs = {
        "measurement_window": int(buffer_window / dt),
        "simul_kwargs": simul_kwargs,
    }

    def __repr__(self):
        return repr(self)


@dataclass
class FarmRowFloris:
    """
    Base Layout.
    Simpe farm with M aligned turbines.
    """

    # Placeholders for parameters as a function of M
    n_turbines = None
    xcoords = None
    ycoords = None

    def get_xcoords(n_turbines):
        return [400 + i * 600 for i in range(n_turbines)]

    def get_ycoords(n_turbines):
        return [400.0 for _ in range(n_turbines)]

    dt = 1
    buffer_window = 1
    t_init = 0

    interface_kwargs = {}
    simul_kwargs = {"xcoords": xcoords, "ycoords": ycoords, "direction": 270}
    interface_kwargs["simul_kwargs"] = simul_kwargs

    def __repr__(self):
        return repr(self)


named_cases_dictionary = {
    "Turb16_TCRWP_": [Farm16TCRWPFastfarm, Farm16TCRWPFloris],
    "Turb3_Row1_": [Farm3Fastfarm, Farm3Floris],
    "Turb6_Row2_": [Farm6Fastfarm, Farm6Floris],
    "Ablaincourt_": [FarmAblaincourtFastfarm, FarmAblaincourtFloris],
}
