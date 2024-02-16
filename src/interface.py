from abc import ABC
from typing import List

import numpy as np
from mpi4py import MPI

# import floris.tools as wfct

YAW_TAG = 0
PITCH_TAG = 1
TORQUE_TAG = 2
POWER_TAG = 3
WINDSPEED_TAG = 4
WINDDIR_TAG = 5


class BaseInterface(ABC):
    def __init__(self):
        self.num_turbines = None

    @property
    def wind_speed(self):
        pass

    @property
    def wind_dir(self):
        pass

    def set_yaw_angles(self, yaws: List):
        pass

    def get_yaw_angles(self) -> List:
        pass

    def get_turbine_powers(self) -> List:
        pass

    def reset_interface(self):
        pass

    def next_wind(self):
        pass


# class FlorisInterface(BaseInterface):
#     def __init__(self, config, layout, wind_series=None):
#         super(FlorisInterface, self).__init__(config)
#         self._layout = layout
#         self.num_turbines = len(layout[0])
#         self.fi = wfct.floris_interface.FlorisInterface(self._config_file)
#         self._wind_series_pnt = 0
#         self._wind_series = None
#         if wind_series is not None:
#             self._wind_series = pd.read_csv(wind_series, header=None)
#             assert self._wind_series.shape[1] == 2  # columns: speed, direction

#     @property
#     def wind_speed(self):
#         return self.fi.floris.farm.wind_map.input_speed[0]

#     @property
#     def wind_dir(self):
#         return self.fi.floris.farm.wind_map.input_direction[0]

#     def next_wind(self):
#         if self._wind_series is not None:
#             speed, direction = self._wind_series.iloc[self._wind_series_pnt]
#             self._wind_series_pnt += 1
#             if self._wind_series_pnt >= self._wind_series.shape[0]:
#                 self._wind_series_pnt = 0
#             self.update_wind(speed, direction)
#         else:
#             speed, direction = self.wind_speed, self.wind_dir
#         return speed, direction

#     def update_wind(self, speed=None, direction=None):
#         speed = self.wind_speed if speed is None else speed
#         direction = self.wind_dir if direction is None else direction
#         self.fi.reinitialize_flow_field(wind_speed=speed, wind_direction=direction)
#         self.fi.calculate_wake()

#     def set_yaw_angles(self, yaws: List, sync: bool = True):
#         # Translate fixed yaw angles (wrt to the west) into relative FLORIS yaw angles
#         # fixed = relative + greedy
#         yaws_relative = [y - self.get_greedy_yaw() for y in yaws]
#         self.fi.floris.farm.set_yaw_angles(yaws_relative)
#         if sync:
#             self.fi.calculate_wake()

#     def get_yaw_angles(self):
#         yaws_relative = self.fi.get_yaw_angles()
#         # Convert relative yaw angles into fixed (wrt to the west)
#         yaws = [less_than_180(y + self.get_greedy_yaw()) for y in yaws_relative]
#         return yaws

#     def get_farm_power(self):
#         return self.fi.get_farm_power()

#     def get_turbine_powers(self) -> List:
#         return self.fi.get_turbine_power()

#     def reset_interface(
#         self,
#         yaws=None,
#         wind_direction=None,
#         wind_speed=None,
#         wind_shear=None,
#         turbulence_intensity=None,
#     ):
#         self.fi.reinitialize_flow_field(
#             layout_array=self._layout,
#             wind_direction=wind_direction,
#             wind_speed=wind_speed,
#             wind_shear=wind_shear,
#             turbulence_intensity=turbulence_intensity,
#         )
#         if yaws is not None:
#             self.set_yaw_angles(yaws)
#         self.fi.calculate_wake()
#         self._wind_series_pnt = 0
#         self.next_wind()


class PowerBuffer:
    def __init__(self, num_turbines: int, size: int = 50_000, agg: str = np.mean):
        self._agg_fn = agg
        self._size = size
        self.pos = -1
        self._buffer = np.zeros((self._size, num_turbines))

    def add(self, measure: np.array):
        if self.pos < self._size - 1:
            self.pos += 1
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)

        self._buffer[self.pos, :] = measure

    def get_last(self):
        return self._buffer[self.pos, :]

    def get_agg(self, window: int = 1):
        start = self.pos - window if self.pos > window else 0
        return self._agg_fn(self._buffer[start : self.pos + 1, :], 0)


class MPI_Interface(BaseInterface):
    def __init__(
        self,
        measurement_window: int = 30,
        buffer_size: int = 50_000,
        num_turbines: int = 3,
        log_file: str = None,
    ):
        super(MPI_Interface, self).__init__()

        # Check communication channels
        self._comm = MPI.COMM_WORLD
        self._buffer_size = buffer_size
        self._measurement_window = measurement_window
        self.num_turbines = num_turbines
        self._power_buffers = PowerBuffer(self.num_turbines, size=self._buffer_size)
        self._wind_buffers = PowerBuffer(2, size=self._buffer_size)
        self.current_yaw_command = np.zeros(num_turbines, dtype=np.double)
        self.current_pitch_command = np.zeros(num_turbines, dtype=np.double)
        self.current_torque_command = np.zeros(num_turbines, dtype=np.double)
        self._logging = False
        if log_file is not None:
            self._log_file = log_file
            self._logging = True

    @property
    def wind_speed(self):
        return self.get_turbine_wind()[0]

    @property
    def wind_dir(self):
        return self.get_turbine_wind()[1]

    def update_command(
        self,
        yaw: np.ndarray = None,
        pitch: np.ndarray = None,
        torque: np.ndarray = None,
    ):
        if yaw is not None:
            self.current_yaw_command = np.radians(yaw.astype(np.double))
        if pitch is not None:
            self.current_pitch_command = np.radians(pitch.astype(np.double))
        if torque is not None:
            self.current_torque_command = torque.astype(np.double)

        self._comm.Send(buf=self.current_yaw_command, dest=0, tag=YAW_TAG)
        self._comm.Send(buf=self.current_pitch_command, dest=0, tag=PITCH_TAG)
        self._comm.Send(buf=self.current_torque_command, dest=0, tag=TORQUE_TAG)
        power, wind = self._wait_for_sim_output()
        self._power_buffers.add(power)
        self._wind_buffers.add(wind)

        if self._logging:
            with open(self._log_file, "a") as fp:
                fp.write(
                    f"Sent command YAW {self.get_yaw_command()} - "
                    f" PITCH {self.get_pitch_command()}"
                    f" TORQUE {self.get_torque_command()}\n"
                    f"***********Received Power: {self.get_turbine_powers()}"
                    f" Wind : {self.get_turbine_wind()}\n"
                )

    def get_yaw_command(self):
        return np.degrees(self.current_yaw_command)

    def get_pitch_command(self):
        return np.degrees(self.current_pitch_command)

    def get_torque_command(self):
        return self.current_torque_command

    def get_farm_power(self):
        powers = self.get_turbine_powers()
        return powers.sum()

    def get_turbine_powers(self, window: int = 1) -> List:
        return self._power_buffers.get_agg(self._measurement_window)

    def get_turbine_wind(self, window: int = 1) -> List:
        return self._wind_buffers.get_agg(self._measurement_window)

    def _wait_for_sim_output(self):
        powers = np.zeros(self.num_turbines, dtype=np.double)
        speeds = np.zeros(self.num_turbines, dtype=np.double)
        directions = np.zeros(self.num_turbines, dtype=np.double)
        self._comm.Recv(powers, source=0, tag=POWER_TAG)
        self._comm.Recv(speeds, source=0, tag=WINDSPEED_TAG)
        self._comm.Recv(directions, source=0, tag=WINDDIR_TAG)
        self._comm.Barrier()

        upstream_point = np.argmax(speeds)
        wspeed = speeds[upstream_point]
        wdir = np.degrees(directions[upstream_point])
        # Keep wind direction positive
        if wdir < 0:
            wdir += 360 - 90
        return powers.astype(np.float32), np.array([wspeed, wdir], dtype=np.float32)

    def reset(self):
        return None
