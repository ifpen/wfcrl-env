"""
Layout of the Total Control Reference Wind Power Plant (TC RWP) (first 15 turbines).
https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
"""

n_turbines = 15

Cx = [(i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(15)]
Cy = [-300 + (i % 4) * 126 * 4 for i in range(15)]

dt = 20
buffer_window = 120

interface_kwargs = {"measurement_window": int(buffer_window / dt)}
