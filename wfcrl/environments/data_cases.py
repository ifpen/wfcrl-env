from dataclasses import dataclass
from typing import Callable, List, Union


def repr(self):
    repr = f"Wind farm simulation on {self.simulator}: "
    repr += f"{self.num_turbines} turbines - {self.max_iter} timesteps\n"
    for arg, val in self.interface_kwargs.items():
        if arg[:2] != "__":
            if isinstance(val, dict):
                repr += f"{arg}: \n"
                for name, param in val.items():
                    repr += f"\t{name}: {param}\n"
            else:
                repr += f"{arg}: {val}\n"
    return repr


@dataclass
class DefaultControl:
    yaw = (-40, 40, 5)
    pitch = (0, 45, 1)
    torque = (-2e4, 2e4, 1e3)


@dataclass
class FarmCase:
    num_turbines: int

    xcoords: Union[List, Callable]
    ycoords: Union[List, Callable]

    dt: int
    buffer_window: int = 300
    t_init: int = 300
    max_iter: int = 100
    set_wind_speed: bool = False
    set_wind_direction: bool = False

    @property
    def interface_kwargs(self):
        return None

    def __repr__(self):
        return repr(self)

    def dict(self):
        return self.interface_kwargs


class FastFarmCase(FarmCase):
    simulator: str = "FastFarm"
    set_wind_speed: bool = False
    set_wind_direction: bool = True

    @property
    def interface_kwargs(self):
        params = {
            "max_iter": self.max_iter,
            "num_turbines": self.num_turbines,
        }
        params.update(self.simul_params)
        return params

    @property
    def avg_window(self):
        return int(self.buffer_window / self.dt)

    @property
    def simul_params(self):
        return {
            "xcoords": self.xcoords,
            "ycoords": self.ycoords,
            "speed": 8,
            "dt": self.dt,
        }


class FlorisCase(FarmCase):
    simulator: str = "Floris"
    set_wind_speed: bool = False
    set_wind_direction: bool = False

    @property
    def interface_kwargs(self):
        return self.simul_params

    @property
    def simul_params(self):
        return {
            "xcoords": self.xcoords,
            "ycoords": self.ycoords,
            "direction": 270,
            "speed": 8,
        }


# 3 turbines row layouts
fastfarm_3t = FastFarmCase(
    num_turbines=3,
    xcoords=[0.0, 504.0, 1008.0],
    ycoords=[0.0, 0.0, 0.0],
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_3t = FlorisCase(
    num_turbines=3,
    xcoords=[0.0, 504.0, 1008.0],
    ycoords=[0.0, 0.0, 0.0],
    dt=60,
    buffer_window=1,
    t_init=0,
)

# 2 x 3 layouts
fastfarm_6t = FastFarmCase(
    num_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[-252, -252, -252, 252, 252, 252],
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_6t = FlorisCase(
    num_turbines=6,
    xcoords=[0.0, 504.0, 1008.0, 0.0, 504.0, 1008.0],
    ycoords=[-252, -252, -252, 252, 252, 252],
    dt=60,
    buffer_window=1,
    t_init=0,
)


# 16 turb layouts
# Used in Bizon Monroc, C., Bušić, A., Dubuc, D., & Zhu, J.
# "Actor critic agents for wind farm control", 2023
# fmt: off
xcoords = [
    891.5, 891.5, 891.5, 2088.6, 2089.1, 2088.2, 3285.7, 3285.3,
    3285.3, 3285.3, 3285.7, 4482.4, 4482.4, 4482.4, 4482.8, 4481.9,
]
# translated at x = 0
xcoords = [xc - min(xcoords) for xc in xcoords]
ycoords = [
    5169.3, 3743.2, 2317.2, 5168.7, 3743.5, 2318.0, 6594.1, 5169.4,
    3743.4, 2317.3, 892.2, 6594.9, 5168.8, 3742.8, 2317.6, 892.1,
]
ycoords = [yc - (max(ycoords) + min(ycoords)) / 2 for yc in ycoords]

# fmt: on
fastfarm_16t = FastFarmCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_16t = FlorisCase(
    num_turbines=16,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# 32 turb layouts
# Used in Bizon Monroc, C., Bušić, A., Dubuc, D., & Zhu, J.
# "Actor critic agents for wind farm control", 2023
# fmt: off
xcoords = [
    891.5, 891.5, 891.5, 2088.6, 2089.1, 2088.2, 3285.7, 3285.3,
    3285.3, 3285.3, 3285.7, 4482.4, 4482.4, 4482.4, 4482.8, 4481.9,
    5679.5, 5679.9, 5679.1, 5679.0, 5679.5, 6876.2, 6876.2, 6876.1,
    6876.6, 6876.6, 8073.2, 8073.2, 8073.7, 8072.8, 8073.3, 9270.8,
]
xcoords = [xc - min(xcoords) for xc in xcoords]
ycoords = [
    5169.3, 3743.2, 2317.2, 5168.7, 3743.5, 2318.0, 6594.1, 5169.4,
    3743.4, 2317.3, 892.2, 6594.9, 5168.8, 3742.8, 2317.6, 892.1,
    6594.2, 5169.1, 3743.5, 2317.5, 892.3, 6595.0, 5169.0, 3742.9,
    2317.7, 891.7, 6594.4, 5168.3, 3743.2, 2317.6, 892.4, 6594.6,
]
ycoords = [yc - (max(ycoords) + min(ycoords)) / 2 for yc in ycoords]
# fmt: on
fastfarm_32t = FastFarmCase(
    num_turbines=32,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_32t = FlorisCase(
    num_turbines=32,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# TCRWPF layouts
#  Layout of the Total Control Reference Wind Power Plant (TC RWP).
# https://farmconners.readthedocs.io/en/latest/provided_data_sets.html
xcoords = [(i // 4) * 126 * 4 + int(i % 2 == 0) * 126 * 2 for i in range(32)]
xcoords = [xc - min(xcoords) for xc in xcoords]
ycoords = [-300 + (i % 4) * 126 * 4 for i in range(32)]
ycoords = [yc - (max(ycoords) + min(ycoords)) / 2 for yc in ycoords]

fastfarm_TCRWP = FastFarmCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_TCRWP = FlorisCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# Ablaincourt layouts
# from the layout of
# T. Duc, O. Coupiac, N. Girard, G. Giebel, and T. G ̈oc ̧men, ""Local
# turbulence parameterization improves the jensen wake model and its
# implementation for power optimization of an operating wind farm"
xcoords = [484.8, 797.1, 1038.8, 1377.6, 1716.9, 2057.3, 2400.0]
ycoords = [274.0, 251.0, 66.9, -22.7, -112.5, -195.3, -259.0]
fastfarm_ablaincourt = FastFarmCase(
    num_turbines=7,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_ablaincourt = FlorisCase(
    num_turbines=7,
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# Horns Rev 1 layout
# fmt: off
xcoords = [
    978.66429328, 1046.94882783, 1114.77605221,
    1182.63478828, 1250.7064236 , 1318.35226044, 1386.42389576,
    1454.70843032, 1536.12485835, 1604.86349156, 1673.35516693,
    1741.84684231, 1810.55337238, 1879.01294462, 1947.93432938,
    2301.66729185, 2369.94411253, 2438.8654973 , 2507.35717267,
    2576.06370274, 2657.48013077, 2725.88455072, 2794.28897067,
    2862.26554207, 2930.638141  , 2998.82863667, 3067.01913235,
    3135.66929759, 3214.94164288, 3283.37788384, 3351.5365585 ,
    3419.94097845, 3488.3453984 , 3556.32196979, 3624.72638974,
    3693.13080969, 3774.54629069, 3842.83000239, 3911.04995735,
    3979.54778283, 4047.79961616, 4115.83733572, 4184.48551824,
    4252.73735157, 4334.1537796 , 4402.77309866, 4470.68734397,
    4539.09283338, 4607.22090834, 4675.59460536, 4743.96830237,
    4812.34199938, 4891.61434467, 4960.35297788, 5028.6619017 ,
    5097.36843177, 5166.28981654, 5234.56663722, 5302.99410631,
    5371.94759422, 5455.50810499, 5524.09563166, 5591.79604732,
    5660.20153672, 5728.54344134, 5796.7033087 , 5865.10879811,
    5933.69632477, 6012.9696171 , 6081.19193378, 6149.56453272,
    6217.60474614, 6286.00916609, 6354.13601973, 6423.00010925,
    6491.15878391
]
xcoords = [xc - min(xcoords) for xc in xcoords]
ycoords = [
    5447.12743106, 4888.84439828, 4334.30025777,
    3779.49848277, 3222.95607889, 2669.89493278, 2113.3525289 ,
    1555.06949612, 5445.55762914, 4888.83332877, 4334.10917624,
    3779.38502372, 3222.9207314 , 2668.45658693, 2110.25215482,
    3777.84262442, 3224.85861169, 2666.65417958, 2111.93002705,
    1555.46573474, 5445.95386775, 4889.4522232 , 4332.95057866,
    3779.92968031, 3223.68691446, 2668.92564301, 2114.16437157,
    1555.66347523, 5446.15085062, 4889.39032738, 4334.88793462,
    3778.38629008, 3221.88464553, 2668.86374719, 2112.36210264,
    1555.8604581 , 5444.58180632, 4889.56895881, 4335.07432872,
    3778.32115554, 3223.56741674, 2670.55400361, 2112.57872218,
    1557.82498337, 5448.31311639, 4889.80853138, 4337.04266434,
    3780.2784761 , 3225.7722123 , 2669.26678789, 2112.76136349,
    1556.25593908, 5446.74331447, 4890.0190141 , 4336.77499331,
    3780.31070099, 3222.10626889, 2669.12225615, 2114.91811974,
    1556.45367957, 5446.94257021, 4888.69674904, 4337.67127877,
    3780.90709053, 3224.66042996, 2669.89540232, 2113.13121408,
    1554.88539291, 5447.13955308, 4892.11940294, 4335.87663709,
    3782.33798136, 3225.83633681, 2671.59282275, 2111.35155331,
    1556.84916056
]
ycoords = [yc - (max(ycoords) + min(ycoords)) / 2 for yc in ycoords]
# fmt: on
fastfarm_hornsrev1 = FastFarmCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_hornsrev1 = FlorisCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)


# Ormonde wind farm layout
# fmt: off
xcoords = [
    4093.62372567192, 4586.23169601483, 4207.909022918064, 3822.671339711839,
    3439.185657229091, 3061.6924881018426, 2676.475469958829, 2295.5609596374716,
    1915.5069508804402, 3868.77479147359, 3479.8198869796765, 3094.605331044565,
    2707.6286696999778, 2315.2792370310417, 1931.7403314094568, 1544.76367006487,
    3525.49397590361, 3133.125437405249, 2740.608840334446, 2350.7386279644306,
    1957.3645790887745, 1564.1385887855513, 1172.5534728058212, 786.11306765524,
    2804.60426320667, 2408.246264201987, 2009.1108340703724, 1611.699404782204,
    1214.816882330398, 819.5326086704703, 422.23540315107
]
ycoords = [
    1127.3791821561301, 1672.9962825278799, 2045.790730432492, 2425.399143133105,
    2803.2811561519584, 3175.258221181207, 3554.846270786033, 3930.194682795655,
    4304.69516728624, 1349.4126394052, 1718.7113456882025, 2084.458725409041,
    2451.879160901989, 2824.400849718425, 3188.557259674342, 3555.9776951672898,
    650.60966542751, 1010.7905655699709, 1371.1073784186349, 1728.9949006129034,
    2090.098824895329, 2451.066836471561, 2810.5285815333623, 3165.26765799256,
    327.02602230483, 677.2402547409571, 1029.9085713956556, 1381.0535943774057,
    1731.7312855475302, 2080.9967951159333, 2432.04089219331
]
# fmt: on
fastfarm_ormonde = FastFarmCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_ormonde = FlorisCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# Horns Rev 2 wind farm layout
# fmt: off
xcoords = [
    5212.666927, 3586.690071, 3038.283676, 2502.705501, 1957.5061600000001,
    1421.927985, 873.52159, 341.150469, 3548.2054120000003, 3003.0060719999997,
    2457.8067309999997, 1922.228556, 1380.236271, 831.829876, 286.630535,
    3551.412467, 3022.2484010000003, 2473.842006, 1928.642666, 1383.443326,
    838.2439860000001, 293.044645, 3609.139456, 3063.9401159999998, 2531.568995,
    1986.369655, 1444.37737, 908.7991939999999, 363.599854, 3708.5581589999997,
    3163.358819, 2627.780643, 2101.823633, 1563.038403, 1017.839062, 488.674997,
    3849.668576, 3310.8833459999996, 2778.5122260000003, 2249.34816, 1720.184095,
    1191.020029, 671.477129, 4019.6424879999995, 3496.892533, 2977.349632, 2454.599676,
    1938.263831, 1428.3420950000002, 905.592139, 4240.929279, 3731.007544,
    3237.1210819999997, 2727.199347, 2214.0705559999997, 1704.14882, 1197.434139,
    4513.5289490000005, 4013.228378, 3516.134862, 3028.662511, 2534.77605, 2037.682534,
    1550.210183, 4824.613279, 4340.347983, 3856.082686, 3378.2315, 2903.587368,
    2432.1502920000003, 1954.299105, 5170.975213000001, 4702.745191, 4253.757498999999,
    3788.734532, 3333.332731, 2868.309764, 2403.2867969999998, 5562.235916, 5119.662334,
    4673.881697, 4231.308115, 3798.355697, 3355.782115, 2910.0014779999997, 5988.774223,
    5565.442971, 5138.904663, 4725.194576, 4305.070378, 3881.739126, 3455.200818
]
ycoords = [
    6415.563578, 1990.3730449999998, 1945.447252, 1897.3124750000002, 1839.550742,
    1794.62495, 1749.6991580000001, 1701.56438, 2667.468913, 2657.841957,
    2648.2150020000004, 2622.543121, 2616.12515, 2612.916165, 2600.080225,
    3376.6546329999996, 3383.072603, 3411.95347, 3440.834336, 3456.8792620000004,
    3488.969114, 3492.178099, 4056.959487, 4105.094264, 4172.482952, 4239.871641,
    4288.006418, 4355.395106, 4403.5298840000005, 4740.473325, 4846.369836,
    4933.012435000001, 5016.446049, 5125.551544, 5212.194144, 5311.672684, 5427.196149,
    5552.34657, 5674.288006000001, 5809.065383, 5934.2158039999995, 6056.1572400000005,
    6178.0986760000005, 6088.247092, 6245.487365, 6412.354593, 6582.430805999999,
    6742.880064, 6906.538307, 7063.77858, 6755.716005, 6945.046128999999, 7140.794224,
    7317.288407999999, 7519.454473, 7708.784597, 7907.741677, 7391.095066,
    7609.306057000001, 7837.144002999999, 8074.608905, 8286.401925, 8520.657842,
    8751.704773, 7994.384276, 8270.357, 8520.657842, 8787.00361, 9034.095467000001,
    9274.769354, 9550.742078000001, 8597.673486, 8889.691135, 9168.872844, 9451.263538,
    9743.281186999999, 10028.88087, 10314.48055, 9181.708784999999, 9483.353389,
    9804.251905, 10131.56839, 10436.421980000001, 10763.73847, 11084.636980000001,
    9720.818291, 10064.1797, 10407.54112, 10750.90253, 11094.263939999999,
    11453.67028, 11800.240670000001
]
# fmt: on
fastfarm_hornsrev2 = FastFarmCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_hornsrev2 = FlorisCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)

# WMR farm layout
# fmt: off
xcoords = [
    6227.98693, 3378.3280896, 4347.000730399999, 5287.63824580002, 6245.1108866000095,
    7196.42677700001, 8167.2210428, 3900.5954734, 4871.9464891999805, 5812.30562940001,
    7721.61578559999, 8675.33167620002, 4425.78448199999, 5359.743622199991,
    8214.491293600011, 9173.842309599999, 4924.781614799999, 5879.854255599999,
    8733.28030220001, 9684.317817600011, 5464.9273735999905, 6427.2000144, 7378.794280199991,
    9287.30443639999, 10234.620326, 5977.03800720001, 6904.0755223999895, 7895.70491359999,
    9810.09344499998, 10763.809336, 6505.9486406, 7463.42128139998, 8404.5804216,
    9378.609812800001, 10340.325703999999, 11288.47672
]
ycoords = [
    5469.043152, 6514.8637897997005, 7132.538836799939, 7762.2424016000205, 8388.74596620019,
    9021.40187620011, 9628.95309560009, 5724.1606004003, 6347.21651039973, 6971.5962475998,
    8218.97485920013, 8849.23076920029, 4993.07617259966, 5589.1227016002795, 7463.8048780002,
    8094.06078799964, 4167.17298319965, 4786.1718573998105, 6654.45403379988, 7282.833771200049,
    3326.01275800027, 3950.6686680000203, 4584.64840519987, 5819.22701679977, 6468.93058160029,
    2533.2427767999598, 3143.1939961997, 3786.74521579973, 5030.92382740013, 5661.1797374003,
    1733.46341459983, 2366.39549720015, 2997.97523460024, 3614.85028140031, 4249.10619139984,
    4870.53358340008
]
# fmt: on
fastfarm_wmr = FastFarmCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=3,
    buffer_window=1,
    t_init=100,
    set_wind_direction=True,
)
floris_wmr = FlorisCase(
    num_turbines=len(xcoords),
    xcoords=xcoords,
    ycoords=ycoords,
    dt=60,
    buffer_window=1,
    t_init=0,
)


class FarmRowFastfarm(FastFarmCase):
    """
    Base Layout.
    Simple farm with M aligned turbines.
    """

    dt = 3
    buffer_window = 1
    t_init = 100
    set_wind_direction = True
    set_wind_speed = False

    @classmethod
    def get_xcoords(cls, num_turbines):
        return [i * 4 * 126.0 for i in range(num_turbines)]

    @classmethod
    def get_ycoords(cls, num_turbines):
        return [0.0 for _ in range(num_turbines)]


class FarmRowFloris(FlorisCase):
    """
    Base Layout.
    Simple farm with M aligned turbines.
    """

    dt = 60
    buffer_window = 1
    t_init = 0
    set_wind_direction = False
    set_wind_speed = False

    @classmethod
    def get_xcoords(cls, num_turbines):
        return [i * 4 * 126.0 for i in range(num_turbines)]

    @classmethod
    def get_ycoords(cls, num_turbines):
        return [0.0 for _ in range(num_turbines)]


named_cases_dictionary = {
    "Turb_TCRWP_": [fastfarm_TCRWP, floris_TCRWP],
    "Turb3_Row1_": [fastfarm_3t, floris_3t],
    "Turb6_Row2_": [fastfarm_6t, floris_6t],
    "Turb16_Row5_": [fastfarm_16t, floris_16t],
    "Turb32_Row5_": [fastfarm_32t, floris_32t],
    "Ablaincourt_": [fastfarm_ablaincourt, floris_ablaincourt],
    "HornsRev1_": [fastfarm_hornsrev1, floris_hornsrev1],
    "HornsRev2_": [fastfarm_hornsrev2, floris_hornsrev2],
    "Ormonde_": [fastfarm_ormonde, floris_ormonde],
    "WMR_": [fastfarm_wmr, floris_wmr],
}
