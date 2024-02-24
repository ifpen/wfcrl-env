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
