import os
from abc import ABC
from typing import List

import numpy as np
from dotenv import load_dotenv
from floris import tools
from mpi4py import MPI

from wfcrl.simul_utils import create_ff_case, create_floris_case

# load environment variables from .env
load_dotenv(override=True)

# default measure indices in the com matrix


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

    def empty(self):
        self._buffer[:] = 0.0
        self.pos = -1


class MPI_Interface(BaseInterface):
    CONTROL_SET = ["yaw", "pitch", "torque"]
    # `wind_measurements` is not read from the simulator
    # but computed by the interface
    DEFAULT_MEASURE_MAP = {
        "wind_speed": 0,
        "power": 1,
        "wind_direction": 2,
        "yaw": 3,
        "pitch": 4,
        "torque": 5,
        "load": [6, 7, 8, 9, 10, 11],
        "wind_measurements": None,
    }
    YAW_TAG = 1
    PITCH_TAG = 2
    TORQUE_TAG = 3
    COM_TAG = 0
    MEASURES_TAG = 4

    def __init__(
        self,
        measurement_window: int = 30,
        buffer_size: int = 50_000,
        num_turbines: int = 3,
        log_file: str = None,
        measure_map: dict = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        target_process_rank: int = None,
        max_iter: int = 500,
    ):
        super().__init__()

        # Check communication channels
        self._comm = comm
        if target_process_rank is None:
            rank = self._comm.Get_rank()
            target_process_rank = 1 - rank
        self._target_process_rank = target_process_rank
        self._buffer_size = buffer_size
        self._measurement_window = measurement_window
        self._num_measures = None
        self.current_measures = None
        self._max_iter = max_iter
        # Maintain mapping with measure name -> ID in measure matrix
        self.measure_map = (
            self.DEFAULT_MEASURE_MAP if measure_map is None else measure_map
        )
        self.num_turbines = num_turbines
        self._power_buffers = PowerBuffer(self.num_turbines, size=self._buffer_size)
        self._wind_buffers = PowerBuffer(2, size=self._buffer_size)
        self._current_yaw_command = np.zeros(num_turbines + 1, dtype=np.double)
        self._current_pitch_command = np.zeros(num_turbines + 1, dtype=np.double)
        self._current_torque_command = np.zeros(num_turbines + 1, dtype=np.double)
        self._num_iter = 0

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
        assert self.current_measures is not None, "Call reset before `update_command`"
        if yaw is not None:
            self._current_yaw_command[1:] = np.radians(yaw.astype(np.double))
            self._current_yaw_command[0] = 1.0
        if pitch is not None:
            self._current_pitch_command[1:] = np.radians(pitch.astype(np.double))
            self._current_pitch_command[0] = 1.0
        if torque is not None:
            self._current_torque_command[1:] = torque.astype(np.double)
            self._current_torque_command[0] = 1.0

        self._comm.Send(
            buf=self._current_yaw_command,
            dest=self._target_process_rank,
            tag=self.YAW_TAG,
        )
        self._comm.Send(
            buf=self._current_pitch_command,
            dest=self._target_process_rank,
            tag=self.PITCH_TAG,
        )
        self._comm.Send(
            buf=self._current_torque_command,
            dest=self._target_process_rank,
            tag=self.TORQUE_TAG,
        )
        power, wind = self._wait_for_sim_output()
        self._power_buffers.add(power)
        self._wind_buffers.add(wind)

        self._num_iter += 1
        if self._num_iter == self._max_iter:
            self._finalize_mpi_comm()

        if self._logging:
            with open(self._log_file, "a") as fp:
                fp.write(
                    f"Sent command YAW {self.get_yaw_command()} - "
                    f" PITCH {self.get_pitch_command()}"
                    f" TORQUE {self.get_torque_command()}\n"
                    f"***********Received Power: {power} - "
                    f"Filtered Power: (window {self._measurement_window}):"
                    f"{self.get_turbine_powers()} - "
                    f" Wind : {self.get_turbine_wind()}\n"
                )

        return self._num_iter == self._max_iter

    def set_comm(self, comm):
        self._comm = comm

    def reset(self):
        self._num_iter = 0
        self._current_yaw_command = np.zeros(self.num_turbines + 1, dtype=np.double)
        self._current_pitch_command = np.zeros(self.num_turbines + 1, dtype=np.double)
        self._current_torque_command = np.zeros(self.num_turbines + 1, dtype=np.double)
        self._power_buffers.empty()
        self._wind_buffers.empty()

        num_measures = np.array([0], dtype=int)
        self._comm.Recv(
            num_measures, source=self._target_process_rank, tag=self.COM_TAG
        )
        self._comm.Send(
            buf=np.array([self._max_iter], dtype=int),
            dest=self._target_process_rank,
            tag=self.COM_TAG,
        )
        self._num_measures = num_measures[0]
        print(
            f"Interface: will receive {self._num_measures} measures at every iteration"
        )
        self.current_measures = (
            np.zeros((self.num_turbines, self._num_measures)) * np.nan
        )

    def _finalize_mpi_comm(self):
        # Disconnect from intercommunicator
        if isinstance(self._comm, MPI.Intercomm):
            self._comm.Disconnect()

    def get_yaw_command(self):
        if 1 - self._current_yaw_command[0]:
            return None
        return np.degrees(self._current_yaw_command).copy()[1:]

    def get_pitch_command(self):
        if 1 - self._current_pitch_command[0]:
            return None
        return np.degrees(self._current_pitch_command).copy()[1:]

    def get_torque_command(self):
        if 1 - self._current_torque_command[0]:
            return None
        return self._current_torque_command.copy()[1:]

    def get_farm_power(self):
        powers = self.get_turbine_powers()
        return powers.sum()

    def get_turbine_powers(self, window: int = 1) -> List:
        return self._power_buffers.get_agg(self._measurement_window)

    def get_turbine_wind(self, window: int = None) -> List:
        if window is None:
            window = self._measurement_window
        return self._wind_buffers.get_agg(window)

    def get_measure(self, measure: str) -> np.ndarray:
        if measure == "wind_measurements":
            return self.get_turbine_wind()
        return self.current_measures[:, self.measure_map[measure]]

    def _wait_for_sim_output(self):
        size_buffer = self.num_turbines * self._num_measures
        measures = np.zeros(size_buffer, dtype=np.double)
        self._comm.Recv(
            measures, source=self._target_process_rank, tag=self.MEASURES_TAG
        )
        self._comm.Barrier()
        measures = measures.reshape((self.num_turbines, self._num_measures))
        # print(f"Received measures matrix from simulator: {measures}")
        speeds = measures[:, self.measure_map["wind_speed"]].flatten()
        directions = measures[:, self.measure_map["wind_direction"]].flatten()
        powers = measures[:, self.measure_map["power"]].flatten()
        upstream_point = np.argmax(speeds)
        wspeed = speeds[upstream_point]
        wdir = np.degrees(directions[upstream_point])
        wdir = wdir - 90
        # Keep wind direction positive
        if wdir < 0:
            wdir = wdir + 360
        self.current_measures = measures
        return powers.astype(np.float32), np.array([wspeed, wdir], dtype=np.float32)


class FastFarmInterface(MPI_Interface):
    def __init__(
        self,
        num_turbines: int,
        simul_kwargs: dict,
        measurement_window: int = 30,
        buffer_size: int = 50_000,
        log_file: str = None,
        measure_map: dict = None,
        max_iter: int = int(1e4),
    ):
        self._path_to_fastfarm_exe = os.getenv("FAST_FARM_EXECUTABLE")
        self._simul_file = create_ff_case(max_iter=max_iter, **simul_kwargs)
        super().__init__(
            measurement_window=measurement_window,
            buffer_size=buffer_size,
            num_turbines=num_turbines,
            log_file=log_file,
            measure_map=measure_map,
            comm=None,
            target_process_rank=0,
            max_iter=max_iter,
        )

    def reset(self):
        print("Spawning process", self._path_to_fastfarm_exe, self._simul_file)
        spawn_comm = MPI.COMM_SELF.Spawn(
            self._path_to_fastfarm_exe, args=[self._simul_file], maxprocs=1
        )
        self.set_comm(spawn_comm)
        super().reset()


class FlorisInterface(BaseInterface):
    CONTROL_SET = ["yaw"]
    # `wind_measurements` handled separately
    DEFAULT_MEASURE_MAP = {"yaw": 0, "wind_measurements": None}
    YAW_TAG = 1
    PITCH_TAG = 2
    TORQUE_TAG = 3
    COM_TAG = 0
    MEASURES_TAG = 4

    def __init__(
        self,
        num_turbines: int,
        simul_kwargs: dict,
        max_iter: int = int(1e4),
        log_file: str = None,
    ):
        super().__init__()

        simul_file = create_floris_case(**simul_kwargs)
        self.num_turbines = num_turbines
        self.fi = tools.FlorisInterface(simul_file)
        self.fi.reinitialize()
        self._current_yaw_command = np.zeros((1, 1, self.num_turbines))
        self.measure_map = self.DEFAULT_MEASURE_MAP
        self.current_measures = (
            np.zeros((self.num_turbines, len(self.measure_map) - 1)) * np.nan
        )
        self._num_iter = 0
        self._max_iter = max_iter
        self._logging = False
        if log_file is not None:
            self._log_file = log_file
            self._logging = True

    @property
    def wind_speed(self):
        return self.fi.floris.flow_field.wind_speeds[0]

    @property
    def wind_dir(self):
        return self.fi.floris.flow_field.wind_directions[0]

    def update_command(
        self,
        yaw: np.ndarray = None,
    ):
        if yaw is not None:
            self._current_yaw_command[0, 0, :] = yaw.astype(np.double)
        self.fi.calculate_wake(yaw_angles=self._current_yaw_command)
        self.current_measures[:, 0] = self.fi.floris.farm.yaw_angles
        self._num_iter += 1
        if self._logging:
            with open(self._log_file, "a") as fp:
                fp.write(
                    f"Sent command YAW {self.get_yaw_command()} - "
                    f"***********Received Power: {self.get_turbine_powers()}"
                    f" Wind : {self.get_turbine_wind()}\n"
                )
        return self._num_iter == self._max_iter

    def reset(self):
        self._num_iter = 0
        self._current_yaw_command = np.zeros((1, 1, self.num_turbines))
        self.current_measures = (
            np.zeros((self.num_turbines, len(self.measure_map) - 1)) * np.nan
        )

    def get_yaw_command(self):
        return self._current_yaw_command.copy().flatten()

    def get_farm_power(self):
        powers = self.get_turbine_powers()
        return powers.sum()

    def get_turbine_powers(self) -> List:
        return self.fi.get_turbine_powers().flatten()

    def get_turbine_wind(self) -> List:
        return np.array([self.wind_speed, self.wind_dir])

    def get_measure(self, measure: str) -> np.ndarray:
        if measure not in self.measure_map:
            return None
        if measure == "wind_measurements":
            return self.get_turbine_wind()
        return self.current_measures[:, self.measure_map[measure]]
