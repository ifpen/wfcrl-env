"""
2 x 3 Layout.
Used on FAST.Farm
"""
n_turbines = 6

Cx = [0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0]
Cy = [-252, -252, -252, 252, 252, 252]

dt = 3
buffer_window = 600
t_init = 9

simul_kwargs = {"Cx": Cx, "Cy": Cy, "dt": dt}
interface_kwargs = {"measurement_window": int(buffer_window / dt)}
