"""
Layout of
T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
turbulence parameterization improves the jensen wake model and its
implementation for power optimization of an operating wind farm"
"""

N_TURBINES = 3

Cx = [400.0, 1000.0, 1600.0]
Cy = [400.0, 400.0, 400.0]

dt = 20
buffer_window = 120

interface_kwargs = {"measurement_window": int(buffer_window / dt)}
