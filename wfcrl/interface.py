import platform
import time
import warnings
from abc import ABC
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from floris import tools
from mpi4py import MPI

from wfcrl.environments import FarmCase
from wfcrl.simul_utils import (
    create_dll,
    create_ff_case,
    create_floris_case,
    get_inflow_file_path,
    read_inflow_info,
    read_simul_info,
    reset_simul_file,
    write_inflow_info,
)


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

    def avg_powers(self) -> List:
        pass

    def init(self):
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

    def get_all(self, window: int = 1):
        start = self.pos - window if self.pos > window else 0
        return self._buffer[start : self.pos + 1, :]

    def get_agg(self, window: int = 1):
        start = self.pos - window if self.pos > window else 0
        return self._agg_fn(self._buffer[start : self.pos + 1, :], 0)

    def empty(self):
        self._buffer[:] = 0.0
        self.pos = -1


class MPI_Interface(BaseInterface):
    CONTROL_SET = ["yaw", "pitch", "torque"]
    YAW_TAG = 1
    PITCH_TAG = 2
    TORQUE_TAG = 3
    COM_TAG = 0
    MEASURES_TAG = 4

    def __init__(
        self,
        measure_map: dict,
        num_turbines: int,
        buffer_size: int = 50_000,
        log_file: str = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
        target_process_rank: int = None,
        max_iter: int = 500,
        default_avg_window: int = 1,
    ):
        super().__init__()

        # Check communication channels
        self._comm = comm
        if target_process_rank is None:
            rank = self._comm.Get_rank()
            target_process_rank = 1 - rank
        self._target_process_rank = target_process_rank
        self._buffer_size = buffer_size
        self._default_avg_window = default_avg_window
        self._num_measures = None
        self.current_measures = None
        self.max_iter = max_iter

        # Validate measure map and store names of measure in array
        self._validate_measure_map(measure_map)
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
        return self.avg_wind()[0]

    @property
    def wind_dir(self):
        return self.avg_wind()[1]

    def _validate_measure_map(self, measure_map):
        inv_measure_map = {}
        for name, indice in measure_map.items():
            if isinstance(indice, int):
                inv_measure_map[indice] = name
            elif isinstance(indice, Iterable):
                for j, indice_i in enumerate(indice):
                    inv_measure_map[indice_i] = f"{name}_{j}"

        assert min(inv_measure_map.keys()) == 0
        assert max(inv_measure_map.keys()) == len(inv_measure_map) - 1
        measure_names = list(inv_measure_map.values())
        self.measure_map = measure_map
        self.measure_names = measure_names

    def update_command(
        self,
        yaw: np.ndarray = None,
        pitch: np.ndarray = None,
        torque: np.ndarray = None,
    ):
        assert self.current_measures is not None, "Call `init` before `update_command`"
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
        if self._num_iter == self.max_iter:
            self._finalize_mpi_comm()

        if self._logging:
            with open(self._log_file, "a") as fp:
                fp.write(
                    f"Sent command YAW {self.get_yaw_command()} - "
                    f" PITCH {self.get_pitch_command()}"
                    f" TORQUE {self.get_torque_command()}\n"
                    f"***********Received Power: {power} - "
                    f"Filtered Power: (window {self._default_avg_window}):"
                    f"{self.avg_powers()} - "
                    f" Wind : {self.avg_wind()}\n"
                )

        return self._num_iter == self.max_iter

    def set_comm(self, comm):
        self._comm = comm

    def init(self):
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
            buf=np.array([self.max_iter], dtype=np.double),
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

    def avg_farm_power(self, window: int = None):
        powers = self.avg_powers(window).squeeze()
        return powers.sum()

    def avg_powers(self, window: int = None) -> List:
        if window is None:
            window = self._default_avg_window
        return np.atleast_1d(self._power_buffers.get_agg(window).squeeze())

    def avg_wind(self, window: int = None) -> List:
        if window is None:
            window = self._default_avg_window
        return self._wind_buffers.get_agg(window).squeeze()

    def last_powers(self, window: int = 0) -> np.ndarray:
        return np.atleast_1d(self._power_buffers.get_all(window).squeeze())

    def last_wind(self, window: int = 0) -> np.ndarray:
        return self._wind_buffers.get_all(window).squeeze()

    def get_measure(self, measure: str) -> np.ndarray:
        if measure == "freewind_measurements":
            return np.atleast_1d(self.last_wind().squeeze())
        return np.atleast_1d(
            self.current_measures[:, self.measure_map[measure]].squeeze()
        )

    def get_all_measures(self) -> Dict:
        df = pd.DataFrame(self.current_measures, columns=self.measure_names)
        # convert angles to degrees
        df[["yaw", "pitch"]] = np.degrees(df[["yaw", "pitch"]])
        return df

    def _wait_for_sim_output(self):
        size_buffer = self.num_turbines * self._num_measures
        measures = np.zeros(size_buffer, dtype=np.double)
        self._comm.Recv(
            measures, source=self._target_process_rank, tag=self.MEASURES_TAG
        )
        self._comm.Barrier()
        measures = measures.reshape((self.num_turbines, self._num_measures))
        # print(f"Received measures matrix from simulator: {measures}")

        # format wind directions - keep wind direction positive
        directions = measures[:, self.measure_map["wind_direction"]].flatten()
        directions = np.degrees(directions) - 90
        directions[directions < 0] = directions[directions < 0] + 360
        measures[:, self.measure_map["wind_direction"]] = directions

        # retrieve wind speeds and power outputs
        speeds = measures[:, self.measure_map["wind_speed"]].flatten()
        powers = measures[:, self.measure_map["power"]].flatten()

        # get upstream point = point of maximum speed
        upstream_point = np.argmax(speeds)
        wspeed = speeds[upstream_point]
        wdir = directions[upstream_point]
        # wdir = np.degrees(directions[upstream_point])
        # wdir = wdir - 90
        # Keep wind direction positive
        # if wdir < 0:
        #     wdir = wdir + 360
        self.current_measures = measures
        return powers.astype(np.float32), np.array([wspeed, wdir], dtype=np.float32)


class FastFarmInterface(MPI_Interface):
    system_type = platform.system().lower()
    if system_type == "windows":
        default_exe_path = "simulators/fastfarm/bin/FAST.Farm_x64_OMP_2023.exe"
    else:
        default_exe_path = "FAST.Farm"
    # `wind_measurements` is not read from the simulator
    # but computed by the interface
    measure_map = {
        "wind_speed": 0,
        "power": 1,
        "wind_direction": 2,
        "yaw": 3,
        "pitch": 4,
        "torque": 5,
        "load": [6, 7, 8, 9, 10, 11],
        "freewind_measurements": None,
    }

    def __init__(
        self,
        num_turbines: int,
        fstf_file: bool,
        buffer_size: int = 50_000,
        log_file: str = None,
        max_iter: int = int(1e4),
        fast_farm_executable: str = default_exe_path,
        default_avg_window: int = 1,
    ):
        self._path_to_fastfarm_exe = fast_farm_executable
        self._simul_file = fstf_file
        self._inflow_file = get_inflow_file_path(fstf_file)
        self._num_resets = 0

        super().__init__(
            default_avg_window=default_avg_window,
            buffer_size=buffer_size,
            num_turbines=num_turbines,
            log_file=log_file,
            measure_map=self.measure_map,
            comm=None,
            target_process_rank=0,
            max_iter=max_iter,
        )

    @classmethod
    def from_file(
        cls,
        fstf_file,
        fast_farm_executable: str = default_exe_path,
        buffer_size: int = 50_000,
        log_file: str = None,
        default_avg_window: int = 1,
    ):
        print(f"Simulation will be started from fstf file {fstf_file}")
        num_turbines, max_iter = read_simul_info(fstf_file)
        print(f"Creating new DLLs for simulation {fstf_file}")
        create_dll(fstf_file)

        return cls(
            num_turbines=num_turbines,
            fstf_file=fstf_file,
            default_avg_window=default_avg_window,
            buffer_size=buffer_size,
            log_file=log_file,
            max_iter=max_iter,
            fast_farm_executable=fast_farm_executable,
        )

    @classmethod
    def from_case(
        cls,
        case: FarmCase,
        fast_farm_executable: str = default_exe_path,
        buffer_size: int = 50_000,
        log_file: str = None,
        output_dir: str = None,
    ):
        if output_dir is None:
            name = f"{case.simulator}__{case.t_init + case.max_iter * case.dt}s"
            name += f"__{case.num_turbines}T_{time.time()}"
            output_dir = f"__simul__/fastfarm/{name}/"

        fstf_file = create_ff_case(
            case.dict(),
            output_dir=output_dir,
        )

        return cls(
            num_turbines=case.num_turbines,
            fstf_file=fstf_file,
            default_avg_window=case.avg_window,
            buffer_size=buffer_size,
            log_file=log_file,
            max_iter=case.max_iter,
            fast_farm_executable=fast_farm_executable,
        )

    def init(self, wind_speed: float = None, wind_direction: float = None):
        # wind speed and direction ignored for now
        if wind_direction is not None:
            warnings.warn(
                f"Wind direction = {wind_direction} requested, but FastFarmInterface"
                "cannot set wind direction in the simulator. Request will be ignored."
            )

        if wind_speed is not None:
            simul_wind_speed = read_inflow_info(self._inflow_file)
            if simul_wind_speed != wind_speed:
                write_inflow_info(self._inflow_file, float(wind_speed))

        simul_file = reset_simul_file(self._simul_file, self._num_resets)
        print("Spawning process", self._path_to_fastfarm_exe, simul_file)
        spawn_comm = MPI.COMM_SELF.Spawn(
            self._path_to_fastfarm_exe, args=[simul_file], maxprocs=1
        )
        self.set_comm(spawn_comm)
        self._num_resets += 1
        super().init()


class FlorisInterface(BaseInterface):
    CONTROL_SET = ["yaw"]
    # `wind_measurements` handled separately

    DEFAULT_MEASURE_MAP = {
        "yaw": 0,
        "wind_speed": 1,
        "wind_direction": 2,
        "load": [3, 4, 5, 6],
        "freewind_measurements": None,
    }

    YAW_TAG = 1
    PITCH_TAG = 2
    TORQUE_TAG = 3
    COM_TAG = 0
    MEASURES_TAG = 4

    def __init__(
        self,
        num_turbines: int,
        simul_file: str,
        max_iter: int = int(1e4),
        log_file: str = None,
        wind_speed: float = None,
        wind_direction: float = None,
        wind_time_series: Union[str, np.ndarray] = None,
    ):
        """
        time_series: np.numpy array or path to csv file
                    first column: speed, second column: direction
        """
        super().__init__()

        self.num_turbines = num_turbines
        self.fi = tools.FlorisInterface(simul_file)
        self.measure_map = self.DEFAULT_MEASURE_MAP
        self._num_measures = sum(
            [
                len(indices) if isinstance(indices, list) else 1
                for indices in self.measure_map.values()
            ]
        )
        self._num_measures -= 1
        self.dt = 60
        self.max_iter = max_iter
        self._logging = False

        # Handle wind as speed / direction time series
        self.wind_time_series = wind_time_series
        self.wind_generator = self._make_wind_generator(
            wind_speed, wind_direction, wind_time_series
        )
        wind_speed, wind_direction = next(self.wind_generator)
        self.init(wind_speed, wind_direction)
        if log_file is not None:
            self._log_file = log_file
            self._logging = True

    def _make_wind_generator(
        self, wind_speed=None, wind_direction=None, time_series=None
    ):
        if time_series is None:

            def wind_generator():
                while True:
                    yield wind_speed, wind_direction

        else:
            if isinstance(time_series, str):
                time_series = pd.read_csv(time_series).values
            assert isinstance(time_series, np.ndarray)
            # start at a random point in the time series
            start = np.random.randint(0, time_series.shape[0])
            time_series = np.r_[time_series[start:], time_series[:start]]

            def wind_generator():
                for ts in time_series:
                    yield ts

        return wind_generator()

    @classmethod
    def from_case(
        cls,
        case: FarmCase,
        log_file: str = None,
        output_dir: str = None,
    ):
        if output_dir is None:
            name = f"{case.simulator}__{case.t_init + case.max_iter * case.dt}s"
            name += f"__{case.num_turbines}T_{time.time()}"
            output_dir = f"__simul__/floris/{name}/"

        simul_file = create_floris_case(case.dict(), output_dir=output_dir)
        return cls(
            num_turbines=case.num_turbines,
            simul_file=simul_file,
            max_iter=case.max_iter,
            log_file=log_file,
            wind_speed=float(case.simul_params["speed"]),
            wind_direction=float(case.simul_params["direction"]),
            wind_time_series=case.simul_params["wind_time_series"],
        )

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
        self.update_wind(*next(self.wind_generator))
        self.fi.calculate_wake(yaw_angles=self._current_yaw_command)
        self.current_measures[
            :, self.measure_map["yaw"]
        ] = self.fi.floris.farm.yaw_angles
        wind_measures_indices = [
            self.measure_map["wind_speed"],
            self.measure_map["wind_direction"],
        ]
        self.current_measures[:, wind_measures_indices] = np.array(
            self.local_wind_measurements()
        ).T
        self.current_measures[:, self.measure_map["load"]] = (
            np.array(self.local_load_proxies()).T * 1e7
        )
        self._num_iter += 1
        if self._logging:
            with open(self._log_file, "a") as fp:
                fp.write(
                    f"Sent command YAW {self.get_yaw_command()} - "
                    f"***********Received Power: {self.avg_powers()}"
                    f" Wind : {self.avg_wind()}\n"
                )
        return self._num_iter == self.max_iter

    def init(self, wind_speed: float = None, wind_direction: float = None):
        if self.wind_time_series and wind_speed is not None:
            warnings.warn(
                f"Wind direction = {wind_speed} requested, but wind_time_series"
                "mode is activated. Request will be ignored."
            )
            wind_speed = None
        if self.wind_time_series and wind_direction is not None:
            warnings.warn(
                f"Wind direction = {wind_direction} requested, but wind_time_series"
                "mode is activated. Request will be ignored."
            )
            wind_direction = None
        self.wind_generator = self._make_wind_generator(
            wind_speed, wind_direction, self.wind_time_series
        )
        self.update_wind(*next(self.wind_generator))
        # self.fi.reinitialize(
        #     wind_speeds=[wind_speed] if wind_speed is not None else None,
        #     wind_directions=[wind_direction] if wind_direction is not None else None,
        # )
        self._num_iter = 0
        self._current_yaw_command = np.zeros((1, 1, self.num_turbines))
        self.current_measures = (
            np.zeros((self.num_turbines, self._num_measures)) * np.nan
        )

    def get_yaw_command(self):
        return self._current_yaw_command.copy().flatten()

    def avg_farm_power(self):
        powers = self.avg_powers()
        return powers.sum()

    def avg_powers(self) -> List:
        return self.fi.get_turbine_powers().flatten()

    def avg_wind(self) -> List:
        """returns average free-stream wind"""
        return np.array([self.wind_speed, self.wind_dir]).squeeze()

    def local_load_proxies(self) -> np.ndarray:
        """we use local turbulences and wind speed variations on the rotors
        as expressed by standard deviation as proxies for loads"""
        turbulences = self.fi.floris.flow_field.turbulence_intensity_field.squeeze()
        # wind speed variations along the 3 axes - standard deviation on rotor speed
        var_u = np.std(self.fi.floris.flow_field.u, (3, 4)).squeeze()
        var_v = np.std(self.fi.floris.flow_field.v, (3, 4)).squeeze()
        var_w = np.std(self.fi.floris.flow_field.w, (3, 4)).squeeze()
        return turbulences, var_u, var_v, var_w

    def local_wind_measurements(self) -> np.ndarray:
        # u (np.array): x-component of velocity.
        # v (np.array): y-component of velocity.
        # w (np.array): z-component of velocity.
        velocities = np.cbrt(np.mean(self.fi.floris.flow_field.u**3, axis=(3, 4)))
        directions = self.wind_dir - np.degrees(
            np.arctan2(self.fi.floris.flow_field.v, self.fi.floris.flow_field.u)
        )
        directions = np.mean(directions, axis=(3, 4))
        return velocities.squeeze(), directions.squeeze()

    def get_measure(self, measure: str) -> np.ndarray:
        if measure not in self.measure_map:
            return None
        if measure == "freewind_measurements":
            return self.avg_wind()
        return self.current_measures[:, self.measure_map[measure]]

    def get_parameters(self):
        pass

    def sample_parameters(self):
        pass

    def update_wind(self, wind_speed: float = None, wind_direction: float = None):
        wind_direction = wind_direction % 360
        if wind_speed != self.wind_speed or wind_direction != self.wind_dir:
            self.fi.reinitialize(
                wind_speeds=[wind_speed] if wind_speed is not None else None,
                wind_directions=[wind_direction]
                if wind_direction is not None
                else None,
            )
