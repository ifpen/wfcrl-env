n_turbines = 16

xcoords = [
    0.8915,
    0.8915,
    0.8915,
    2.0886,
    2.0891,
    2.0882,
    3.2857,
    3.2853,
    3.2853,
    3.2853,
    3.2857,
    4.4824,
    4.4824,
    4.4824,
    4.4828,
    4.4819,
]
xcoords = [x * 1e3 for x in xcoords]

ycoords = [
    5.1693,
    3.7432,
    2.3172,
    5.1687,
    3.7435,
    2.3180,
    6.5941,
    5.1694,
    3.7434,
    2.3173,
    0.8922,
    6.5949,
    5.1688,
    3.7428,
    2.3176,
    0.8921,
]
ycoords = [y * 1e3 for y in ycoords]

dt = 20
buffer_window = 120

interface_kwargs = {"measurement_window": int(buffer_window / dt)}