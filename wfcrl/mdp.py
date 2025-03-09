import copy
from collections import OrderedDict
from typing import Dict, Iterable, Type, Union
from warnings import warn

import numpy as np
from gymnasium import spaces

from wfcrl.environments import FarmCase
from wfcrl.interface import BaseInterface, MPI_Interface


def clip_to_dict_space(element: dict, space: spaces.Dict):
    for name, value in element.items():
        element[name] = np.clip(value, space[name].low, space[name].high)
    return element


class WindFarmMDP:
    """
    Implements the underlying MDP of the wind farm
    States: [velocity, direction, gamma_1, ..., gamma_M] vector of yaws
    Actions: discrete or continous

    controls: dictionary with {name of control from CONTROL_SET: bounds on actuator}
    start_iter: iter at which to start control


    controls example:
        {
            "yaw": (-20, 20, 2),
            "pitch": (-10, 10, 1)
        }
    """

    CONTROL_SET = ["yaw", "pitch", "torque"]
    POSSIBLE_STATE_ATTRIBUTES = [
        "freewind_measurements",
        "wind_speed",
        "wind_direction",
        "yaw",
        "pitch",
        "torque",
    ]
    DEFAULT_BOUNDS = {
        "wind_speed": [3, 28],
        "wind_direction": [0, 360],
        "yaw": [-40, 40],
        "pitch": [0, 360],
        "torque": [-1e5, 1e5],
    }
    ACTUATORS_RATE = {"yaw": 0.3, "pitch": 8}

    def __init__(
        self,
        interface: Union[BaseInterface, Type[BaseInterface]],
        farm_case: FarmCase,
        controls: dict,
        continuous_control: bool = True,
        start_iter: int = 0,
        horizon: int = int(1e6),
    ):
        farm_case.max_iter = horizon
        # if an interface is already instantiated
        if isinstance(interface, BaseInterface):
            self.interface = interface
            warn(
                "Interface already instantiated."
                "Simulation arguments from `Farm case` will be ignored."
            )
        elif interface == MPI_Interface:
            # MPI_Interface connects to an already running
            # process so it does not accept
            # simulation configuration
            # Log warning here ?
            interface_kwargs = farm_case.interface_kwargs
            for key in farm_case.simul_params:
                del interface_kwargs[key]
            self.interface = interface(**interface_kwargs)
        else:
            path_to_simulator = farm_case.interface_kwargs.get("path_to_simulator", None)
            interface_args = (
                [farm_case] 
                if path_to_simulator is None 
                else [farm_case, path_to_simulator]
            )
            self.interface = interface.from_case(*interface_args)
        self.num_turbines = farm_case.num_turbines
        self.continuous_control = continuous_control
        self.horizon = horizon
        self.start_iter = start_iter
        self.farm_case = farm_case

        # Check validity of controls
        self._check_controls(controls)
        self.controls = controls
        self.num_controls = len(controls)
        # All non controlled observations are measured
        # but only if they can be measured by the interface
        self.measures = [
            obs
            for obs in self.POSSIBLE_STATE_ATTRIBUTES
            if (obs not in controls) and (obs in self.interface.measure_map)
        ]
        self.state_attributes = list(self.controls.keys()) + self.measures

        # Setup actions
        if self.continuous_control:
            self.action_space = spaces.Dict(
                {
                    name: spaces.Box(
                        -bounds_and_step[2],
                        bounds_and_step[2],
                        shape=(self.num_turbines,),
                    )
                    for name, bounds_and_step in self.controls.items()
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    name: spaces.MultiDiscrete(
                        # 3 = down, no change, up
                        [3 for _ in range(self.num_turbines)],
                    )
                    for name, bounds_and_step in self.controls.items()
                }
            )

        # Setup state space
        state_space_dict = OrderedDict()
        bound_array = np.ones(self.num_turbines, dtype=np.float32)
        low_ws, high_ws = self.DEFAULT_BOUNDS["wind_speed"]
        (
            low_wd,
            high_wd,
        ) = self.DEFAULT_BOUNDS["wind_direction"]
        for attr in self.state_attributes:
            if attr == "freewind_measurements":
                low = np.array([low_ws, low_wd], dtype=np.float32)
                high = np.array([high_ws, high_wd], dtype=np.float32)
            elif attr in controls:
                low = bound_array * controls[attr][0]
                high = bound_array * controls[attr][1]
            else:
                low = bound_array * self.DEFAULT_BOUNDS[attr][0]
                high = bound_array * self.DEFAULT_BOUNDS[attr][1]
            state_space_dict[attr] = spaces.Box(
                low,
                high,
                shape=low.shape,
            )
        self.state_space = spaces.Dict(state_space_dict)
        self.start_state = None

        # Set up constraints on actuation dict
        self._actuation_accumulator = {
            control: np.zeros(self.num_turbines, dtype=np.float32)
            for control in controls
        }

    def get_state_powers(self):
        return self.interface.avg_powers()

    def get_accumulated_actions(self, agent=None):
        return self._actuation_accumulator.copy()

    def _cast_dict_array(self, state):
        state_cast = OrderedDict()
        for attr, value in state.items():
            state_cast[attr] = value.astype(np.float32)
        return state_cast

    def _check_controls(self, control_dict: Dict):
        for name, bounds_and_step in control_dict.items():
            # Check that the chosen interface implements the controls
            if name not in self.CONTROL_SET:
                raise ValueError(
                    f"Cannot control {name}. Allowed controls are {self.CONTROL_SET}"
                )
            if name not in self.interface.CONTROL_SET:
                raise ValueError(
                    f"Cannot control `{name}`. Interface {self.interface.__class__.__name__}"
                    f" only allows for the following: {self.interface.CONTROL_SET}"
                )
            len_b = len(bounds_and_step)
            if not (
                isinstance(bounds_and_step, Iterable) and len_b >= 2 and len_b <= 3
            ):
                raise TypeError(
                    f"Wrong bounds for actuator {name}:"
                    "Bounds on actuators must be an iterable of the type"
                    " [lower_bound, upper_bound] if control is continuous and"
                    " [lower_bound, upper_bound, step_size] otherwise"
                )
            if not (bounds_and_step[0] < bounds_and_step[1]):
                raise ValueError(
                    f"Wrong bounds for actuator {name}: ensure that"
                    " lower_bound < upper_bound"
                )
            if len_b == 2:
                control_dict[name] = bounds_and_step + (1,)
                warn(
                    f"No step size was provided for actuator {name}. Step size will default to 1."
                )
            if not self.continuous_control:
                if (len_b == 3) and bounds_and_step[2] <= 0:
                    raise ValueError(
                        f"Invalid step size provided for actuator {name}"
                        " the step size must be stricly positive"
                    )

    def _check_state(self, state: Dict):
        for attr, value in state.items():
            if attr not in self.state_attributes:
                raise ValueError(
                    f"Unknwon attribute {attr} in state dict."
                    f"Accepted attributed are: {self.state_attributes}"
                )
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"State attribute {attr} must be a numpy array."
                    f"Received {type(value)}"
                )
            if attr != "freewind_measurements" and not (
                value.shape == (self.num_turbines,)
            ):
                raise TypeError(
                    f"State attribute {attr} must be of shape (NUM_TURBINES,),"
                    f"but received {value.shape}. NUM_TURBINES = {self.num_turbines})"
                )

    def reset(self, seed: int = None, options: dict = None):
        # sample wind_speed and wind_direction
        rng = np.random.default_rng(seed)
        wind_speed, wind_direction = None, None
        if (options is not None) and "wind_speed" in options:
            wind_speed = options["wind_speed"]
        elif not (
            self.farm_case.set_wind_speed or bool(self.farm_case.wind_time_series)
        ):
            wind_speed = 8 * rng.weibull(8)
            wind_speed = np.clip(
                wind_speed,
                self.state_space["freewind_measurements"].low[0],
                self.state_space["freewind_measurements"].high[0],
            )
        if (options is not None) and "wind_direction" in options:
            wind_direction = options["wind_direction"]
        elif not (
            self.farm_case.set_wind_direction or bool(self.farm_case.wind_time_series)
        ):
            wind_direction = rng.normal(270, 20) % 360
            wind_direction = np.clip(
                wind_direction,
                self.state_space["freewind_measurements"].low[1],
                self.state_space["freewind_measurements"].high[1],
            )

        self.interface.init(wind_speed, wind_direction)
        for _ in range(self.start_iter + 1):
            self.interface.update_command()
        start_state = OrderedDict(
            {attr: self.interface.get_measure(attr) for attr in self.state_attributes}
        )
        self.start_state = clip_to_dict_space(start_state, self.state_space)
        self._actuation_accumulator = {
            control: np.zeros(self.num_turbines, dtype=np.float32)
            for control in self.controls
        }
        return self.start_state

    def step_interface(self, state: Dict):
        step_dict = OrderedDict()
        for control in self.controls:
            step_dict[control] = state[control]
        done = self.interface.update_command(**step_dict)
        powers = self.get_state_powers()
        for measure in self.measures:
            state[measure] = self.interface.get_measure(measure)
        loads = self.interface.get_measure("load")
        if loads is not None:
            loads /= 1e7
        return state, powers / 1e6, loads, done

    def take_action(self, state: Dict, joint_action: Dict):
        next_state = self.get_controlled_state_transition(state, joint_action)
        next_state, powers, loads, done = self.step_interface(next_state)
        return next_state, powers, loads, done

    def get_controlled_state_transition(self, state: Dict, joint_action: Dict):
        # Deterministic transition
        if not isinstance(joint_action, dict):
            raise TypeError("Joint action must be a dictionary")
        state = clip_to_dict_space(self._cast_dict_array(state), self.state_space)
        next_state = copy.deepcopy(state)
        for control, command_joint_action in joint_action.items():
            assert control in self.controls, f"Control of `{control}` is not activated"
            command_joint_action = np.array(command_joint_action, np.float32)
            if self.continuous_control:
                command_joint_action = np.clip(
                    command_joint_action,
                    self.action_space[control].low,
                    self.action_space[control].high,
                )
            else:
                # 0, 1, 2 => DOWN, NO CHANGE, UP
                command_joint_action = (command_joint_action - 1) * self.controls[
                    control
                ][-1]
            next_state[control] = np.clip(
                state[control] + command_joint_action,
                self.state_space[control].low,
                self.state_space[control].high,
            )
            # track actuators trajectory for penalization purpose
            if control in self._actuation_accumulator:
                self._actuation_accumulator[control] += np.abs(command_joint_action)
        return next_state
