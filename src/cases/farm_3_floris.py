"""
Base Layout.
Simpe farm with 3 aligned turbines.
"""

n_turbines = 3

xcoords = [400.0, 1000.0, 1600.0]
ycoords = [400.0, 400.0, 400.0]

dt = 1
buffer_window = 1

interface_kwargs = {}
interface_kwargs["layout"] = (xcoords, ycoords)
interface_kwargs["wind_series"] = None
