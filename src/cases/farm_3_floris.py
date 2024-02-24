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
