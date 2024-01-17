from abc import ABC
import json
import time
from typing import List, Tuple

import floris.tools as wfct
import gym
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path

from qlearning_utils import less_than_180

class BaseInterface(ABC):
    def __init__(self, config):
        self._config_file = Path(config)
        self.num_turbines = None
        self.yaw_referential_origin = 270

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

    def get_greedy_yaw(self):
        # Extract absolute yaw wrt origin of the referential
        return less_than_180(self.wind_dir - self.yaw_referential_origin)

class FlorisInterface(BaseInterface):
    def __init__(self, config, layout, wind_series = None):
        super(FlorisInterface, self).__init__(config)
        self._layout = layout
        self.num_turbines = len(layout[0])
        self.fi = wfct.floris_interface.FlorisInterface(self._config_file)
        self._wind_series_pnt = 0
        self._wind_series = None
        if wind_series is not None:
            self._wind_series = pd.read_csv(wind_series, header=None)
            assert self._wind_series.shape[1] == 2 #columns: speed, direction
        
    @property
    def wind_speed(self):
        return self.fi.floris.farm.wind_map.input_speed[0]
    
    @property
    def wind_dir(self):
        return self.fi.floris.farm.wind_map.input_direction[0]

    def next_wind(self):
        if self._wind_series is not None:
            speed, direction = self._wind_series.iloc[self._wind_series_pnt]
            self._wind_series_pnt += 1
            if self._wind_series_pnt >= self._wind_series.shape[0]:
                self._wind_series_pnt = 0
            self.update_wind(speed, direction)
        else:
            speed, direction = self.wind_speed, self.wind_dir
        return speed, direction
    
    def update_wind(self, speed=None, direction=None):
        speed = self.wind_speed if speed is None else speed
        direction = self.wind_dir if direction is None else direction
        self.fi.reinitialize_flow_field(wind_speed=speed, wind_direction=direction)
        self.fi.calculate_wake()
    
    def set_yaw_angles(self, yaws: List, sync: bool = True):
        # Translate fixed yaw angles (wrt to the west) into relative FLORIS yaw angles
        # fixed = relative + greedy
        yaws_relative = [y - self.get_greedy_yaw() for y in yaws]
        self.fi.floris.farm.set_yaw_angles(yaws_relative)
        if sync:
            self.fi.calculate_wake()

    def get_yaw_angles(self):
        yaws_relative = self.fi.get_yaw_angles()
        # Convert relative yaw angles into fixed (wrt to the west)
        yaws = [less_than_180(y + self.get_greedy_yaw()) for y in yaws_relative]
        return yaws

    def get_farm_power(self):
        return self.fi.get_farm_power()

    def get_turbine_powers(self) -> List:
        return self.fi.get_turbine_power()
    
    def reset_interface(self, 
            yaws=None, 
            wind_direction=None, 
            wind_speed=None, 
            wind_shear=None, 
            turbulence_intensity=None
        ):  
        self.fi.reinitialize_flow_field(
            layout_array=self._layout,
            wind_direction=wind_direction, 
            wind_speed=wind_speed,
            wind_shear=wind_shear,
            turbulence_intensity=turbulence_intensity
        )
        if yaws is not None:
            self.set_yaw_angles(yaws)
        self.fi.calculate_wake()
        self._wind_series_pnt = 0
        self.next_wind()


class PowerBuffer:
    def __init__(self, n_turbines: int, size: int = 50_000, agg: str = np.mean):
        self._agg_fn = agg
        self._size = size
        self.pos = -1
        self._buffer = np.zeros((self._size, n_turbines))

    def add(self, measure: np.array):
        if self.pos < self._size - 1:
            self.pos += 1
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)

        self._buffer[self.pos,:] = measure

    def get_last(self):
        return self._buffer[self.pos, :]

    def get_agg(self, window: int = 1):
        start = self.pos - window if self.pos > window else 0
        return self._agg_fn(self._buffer[start:self.pos+1,:], 0)
        

class WindFarmInterface(BaseInterface):
    def __init__(self, 
            config: str,
            measurement_window: int = 30, 
            buffer_size: int = 50_000,
            num_try_logs: int = 50
        ):
        super(WindFarmInterface, self).__init__(config)

        # Read config file
        with self._config_file.open('r') as file:
            self.config = json.load(file)
        
        # Check communication channels
        assert (
                ("com_file_yaws" in self.config)
            and ("com_file_power" in self.config)
            and ("com_file_wind" in self.config)
        )

        # Check validity of communication channels
        self._file_yaws = Path(self.config["com_file_yaws"])
        self._file_power = Path(self.config["com_file_power"])
        self._file_wind = Path(self.config["com_file_wind"])
        
        assert(
            self._file_yaws.exists()
            and self._file_power.exists()
            and self._file_wind.exists()
        )
        # self._file_yaws.touch(exist_ok=True)
        # self._file_power.touch(exist_ok=True)
        # self._file_wind.touch(exist_ok=True)

        self._iter_yaws = 0
        self._buffer_size = buffer_size
        self._measurement_window = measurement_window
        self._num_try_logs = num_try_logs
        self.num_turbines = len(self.config["Wp"]["turbine"]["Crx"])

        self._yaws = self.config["yaw_init"]
        self._log_iter_pnt = None
        self._power_buffers = None
        self._wind_buffers = None
        self._num_wind_stats = None

    @property
    def wind_speed(self):
        if self._wind_buffers is None:
            raise NotImplementedError(
                "Interface must be initialized to access wind information"
            )
        return self.get_turbine_wind()[0]
    
    @property
    def wind_dir(self):
        if self._wind_buffers is None:
            raise NotImplementedError(
                "Interface must be initialized to access wind information"
            )
        if self._num_wind_stats < 2:
            return np.arctan(
                self.config["Wp"]["site"]["v_Inf"] / self.config["Wp"]["site"]["u_Inf"]
            )
        return self.get_turbine_wind()[1]

    def set_yaw_angles(self, yaws: List, sync: bool = True):
        self._yaws = yaws
        if sync:
            self._iter_yaws +=1
            log = np.c_[
                [[self._iter_yaws]], # num changes
                np.array(yaws)[None,:], # new yaws
                [[0]] # read signal
                ]
            np.savetxt(self._file_yaws, log)
            power, wind = self._read_WF_output()
            while self._log_iter_pnt < self._iter_yaws:
                time.sleep(0.05)
                power, wind = self._read_WF_output()
            self._power_buffers.add(power)
            self._wind_buffers.add(wind)
            # TODO (maybe) add error handle in case both read at the same time

    def get_yaw_angles(self):
        return self._yaws

    def get_farm_power(self):
        powers = self.get_turbine_powers()
        return powers.sum()

    def get_turbine_powers(self, window: int = 1) -> List:
        return self._power_buffers.get_agg(self._measurement_window)
    
    def get_turbine_wind(self, window: int = 1) -> List:
        return self._wind_buffers.get_agg(self._measurement_window)

    def _read_WF_output(self):
        # Read simulator logs
        success = False
        read_attempts = self._num_try_logs
        error = None
        while (not success) and (read_attempts > 0):
            try:
                power_log = pd.read_csv(self._file_power, delimiter=" ", header=None)
                if power_log.iloc[:, 1:-1].shape[1] != self.num_turbines:
                    raise AssertionError
                wind_log = pd.read_csv(self._file_wind, delimiter=" ", header=None)
                success = True
            except Exception as e:
                read_attempts -= 1
                error = e
                time.sleep(0.05)
        if not success:
            raise IOError(
                f'Could not read logs at {self._file_power} and {self._file_wind}.'
                'WindFarm Simulator must be running and logging in csv files'
                f'{error}'
            )
        powers = power_log.iloc[:, 1:-1]
        wind = wind_log.iloc[:, 1:-1]
        self._log_iter_pnt = power_log.iloc[0,0]
        return powers.values, wind.values
    
    def reset_interface(self, 
            yaws=None, 
            wind_direction=None, 
            wind_speed=None, 
            wind_shear=None, 
            turbulence_intensity=None
        ):
        self._iter_yaws = 0

        # Init buffers
        power, wind = self._read_WF_output()
        self._power_buffers = PowerBuffer(self.num_turbines, size=self._buffer_size)
        self._wind_buffers = PowerBuffer(wind.shape[1], size=self._buffer_size)
        self._power_buffers.add(power)
        self._wind_buffers.add(wind)
        self._num_wind_stats = np.size(self.get_turbine_wind())
        
        if yaws is not None:
            self._yaws = yaws
            self.set_yaw_angles(yaws)

class FastFarmInterface(WindFarmInterface):
    def __init__(self, 
            config: str,
            ff_out_file: str,
            measurement_window: int = 30, 
            buffer_size: int = 50_000,
            num_try_logs: int = 300,
            start_read_outfile: int = 2,
            wind_aggregation: str = "first"
        ):
        super(FastFarmInterface, self).__init__(
            config, measurement_window, buffer_size, num_try_logs
        )
        """
        wind_aggregation: how to retrieve wind data 
            from point measurements
        """
        self._ff_out_file = Path(ff_out_file)
        assert self._ff_out_file.exists()
        assert wind_aggregation in ["first", "highest"]
        self._wind_aggreg_kind = wind_aggregation
        self._wind_columns = pd.read_csv(
            self._ff_out_file, skiprows=6, 
            nrows=2, sep="\t", header=None
        ).iloc[0].str.strip().values
        self._num_points = int((self._wind_columns.size-1)/3)
        self._ff_out_file_skiprows = 8
        self._names = self._wind_columns[1:]
        self._time = -np.inf
        self._log_iter_pnt = -np.inf
        self._start_read_outfile = start_read_outfile
        # self._time_offset = wind_logs.iloc[-1, "Time"]
        # self._dtime = wind_logs.iloc[-1, "Time"] - wind_logs.iloc[-2, "Time"]

    def _extract_wind(self, logs):
        speeds = np.array([
            np.sqrt((logs.iloc[(3*i):(3*i+2)]**2).sum(0))
            for i in range(self._num_points)
        ])
        if self._wind_aggreg_kind == "first":
            upstream_point = 0
        else: 
            upstream_point = np.argmax(speeds)
        direction = -np.degrees(np.arctan2(
            *logs.iloc[(3*upstream_point):(3*upstream_point+2)]
        ))
        if direction < 0:
            direction += 360
        return np.array([[speeds[upstream_point], direction]])

    def _read_WF_output(self):
        # Read simulator logs
        success = False
        read_attempts = self._num_try_logs
        error = None
        while (not success) and (read_attempts > 0):
            try:
                power_log = pd.read_csv(self._file_power, delimiter=" ", header=None)
                if power_log.iloc[:, 1:-1].shape[1] != self.num_turbines:
                    raise AssertionError(
                        f"Number of rows in log file {self._file_power} does not match number of turbines"
                    )
                if self._log_iter_pnt > self._start_read_outfile:
                    wind_log = pd.read_csv(
                        self._ff_out_file, 
                        skiprows=self._ff_out_file_skiprows+self._log_iter_pnt-2, 
                        sep="\t", header=None
                    ).iloc[-1,:]
                success = True
            except Exception as e:
                if isinstance(e, AssertionError):
                    raise e
                read_attempts -= 1
                error = e
                time.sleep(0.05)
        if not success:
            raise IOError(
                f'Could not read logs at {self._file_power} and {self._file_wind}.'
                'WindFarm Simulator must be running and logging in csv files'
                f'{error}'
            )
        powers = power_log.iloc[:, 1:-1]
        if self._log_iter_pnt > self._start_read_outfile:
            wind = self._extract_wind(wind_log[1:])
            self._time = wind_log.iloc[0]
        else:
            wind = np.array([[8,270]])
        self._log_iter_pnt = power_log.iloc[0,0]
        # self._time = self._time_offset + self._log_iter_pnt * self._dtime
        # wind_log = wind_logs.loc[wind_logs["Time"] == self._time]
        return powers.values, wind