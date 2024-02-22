"""
Layout of
T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
turbulence parameterization improves the jensen wake model and its
implementation for power optimization of an operating wind farm"
"""

N_TURBINES = 7

Cx = [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0]
Cy = [933.0, 910.0, 725.9, 636.3, 546.5, 463.7, 400.0]

dt = 20
buffer_window = 120

interface_kwargs = {"measurement_window": int(buffer_window / dt)}
