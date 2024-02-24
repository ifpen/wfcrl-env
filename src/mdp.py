import copy
from collections import OrderedDict
from typing import Dict, Iterable

import numpy as np
from gymnasium import spaces

from src.interface import BaseInterface


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
    STATE_ATTRIBUTES = ["wind_measurements", "yaw", "pitch", "torque"]
    DEFAULT_BOUNDS = {
        "wind_speed": [3, 28],
        "wind_direction": [0, 360],
        "yaw": [-40, 40],
        "pitch": [0, 360],
        "torque": [-1e5, 1e5],
    }

    def __init__(
        self,
        interface: BaseInterface,
        num_turbines: int,
        controls: dict,
        continuous_control: bool = True,
        interface_kwargs: dict = {},
        start_iter: int = 0,
        horizon: int = int(1e6),
    ):
        interface_kwargs["num_turbines"] = num_turbines
        interface_kwargs["max_iter"] = horizon
        self.interface = interface(**interface_kwargs)
        self.num_turbines = num_turbines
        self.continuous_control = continuous_control
        self.horizon = horizon

        # Check validity of controls
        self._check_controls(controls)
        self.controls = controls
        self.num_controls = len(controls)
        # All non controlled observations are measured
        self.measures = [obs for obs in self.STATE_ATTRIBUTES if obs not in controls]

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
                        start=[-1 for _ in range(self.num_turbines)],
                    )
                    for name, bounds_and_step in self.controls
                }
            )

        # Take first steps in the interface until start_iter
        for t in range(start_iter + 1):
            self.interface.update_command()
        # Retrieve state at start_iter
        start_state = OrderedDict(
            {attr: self.interface.get_measure(attr) for attr in self.STATE_ATTRIBUTES}
        )
        print(f"Start state {start_state}")

        # Setup state space
        state_space_dict = {}
        bound_array = np.ones(num_turbines, dtype=np.float32)
        for attr in self.STATE_ATTRIBUTES:
            if attr == "wind_measurements":
                low_ws, high_ws = self.DEFAULT_BOUNDS["wind_speed"]
                (
                    low_wd,
                    high_wd,
                ) = self.DEFAULT_BOUNDS["wind_direction"]
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
        self.start_state = clip_to_dict_space(start_state, self.state_space)
        print(f"Init MDP with start state {self.start_state}")

    def get_state_powers(self):
        return self.interface.get_turbine_powers()

    def _cast_dict_array(self, state):
        state_cast = {}
        for attr, value in state.items():
            state_cast[attr] = value.astype(np.float32)
        return state_cast

    def _check_controls(self, control_dict: Dict):
        for name, bounds_and_step in control_dict.items():
            if name not in self.CONTROL_SET:
                raise ValueError(
                    f"Cannot control {name}. Allowed controls are {self.CONTROL_SET}"
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
                control_dict[name] = bounds_and_step + [1]
                raise Warning(
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
            if attr not in self.STATE_ATTRIBUTES:
                raise ValueError(
                    f"Unknwon attribute {attr} in state dict."
                    f"Accepted attributed are: {self.STATE_ATTRIBUTES}"
                )
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"State attribute {attr} must be a numpy array."
                    f"Received {type(value)}"
                )
            if attr != "wind_measurements" and not (
                value.shape == (self.num_turbines,)
            ):
                raise TypeError(
                    f"State attribute {attr} must be of shape (NUM_TURBINES,),"
                    f"but received {value.shape}. NUM_TURBINES = {self.num_turbines})"
                )

    def step_interface(self, state: Dict):
        step_dict = {}
        for control in self.controls:
            step_dict[control] = state[control]
        done = self.interface.update_command(**step_dict)
        powers = self.interface.get_turbine_powers()
        for measure in self.measures:
            state[measure] = self.interface.get_measure(measure)
        loads = self.interface.get_measure("load")
        return state, powers / 1e6, loads / 1e6, done

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
                command_joint_action *= self.controls[control][-1]
            next_state[control] = np.clip(
                state[control] + command_joint_action,
                self.state_space[control].low,
                self.state_space[control].high,
            )
        return next_state
