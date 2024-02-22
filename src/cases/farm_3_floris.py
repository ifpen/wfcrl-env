"""
Base Layout.
Simpe farm with 3 aligned turbines.
"""

N_TURBINES = 3

Cx = [400.0, 1000.0, 1600.0]
Cy = [400.0, 400.0, 400.0]

dt = 1
buffer_window = 1

interface_kwargs = {}
interface_kwargs["layout"] = (Cx, Cy)
interface_kwargs["wind_series"] = None
