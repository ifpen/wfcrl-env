"""
Base Layout.
Simpe farm with 3 aligned turbines.
"""

n_turbines = 3

Cx = [0.0, 504.0, 1008.0]
Cy = [0.0, 0.0, 0.0]

dt = 3
buffer_window = 600
t_init = 9  # 100

simul_kwargs = {"Cx": Cx, "Cy": Cy, "dt": dt}
interface_kwargs = {"measurement_window": int(buffer_window / dt)}
