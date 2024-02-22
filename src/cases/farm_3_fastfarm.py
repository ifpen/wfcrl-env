"""
Base Layout.
Simpe farm with 3 aligned turbines.
"""

N_TURBINES = 3

Cx = [0.0, 504.0, 1008.0]
Cy = [0.0, 0.0, 0.0]

dt = 3
buffer_window = 600
t_init = 100

interface_kwargs = {"measurement_window": int(buffer_window / dt)}
