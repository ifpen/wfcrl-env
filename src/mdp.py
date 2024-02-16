import copy
from collections import OrderedDict
from typing import Dict, Iterable

import numpy as np
from gymnasium import spaces

from src.interface import BaseInterface


class WindFarmMDP:
    """
    Implements the underlying MDP of the wind farm
    States: [velocity, direction, gamma_1, ..., gamma_M] vector of yaws
    Actions: discrete or continous

    controls: dictionary with {name of control from CONTROL_SET: bounds on actuator}


    controls example:
        {
            "yaw": (-20, 20, 2),
            "pitch": (-10, 10, 1)
        }
    """

    CONTROL_SET = ["yaw", "pitch", "torque"]
    STATE_ATTRIBUTES = ["wind_measurements", "yaw", "pitch", "torque"]

    def __init__(
        self,
        interface: BaseInterface,
        num_turbines: int,
        controls: dict,
        start_state: OrderedDict,
        continuous_control: bool = True,
        interface_kwargs: dict = {},
    ):
        interface_kwargs["num_turbines"] = num_turbines
        self.interface = interface(**interface_kwargs)
        self.num_turbines = num_turbines
        self.continuous_control = continuous_control

        # Check validity of controls
        self._check_controls(controls)
        self.controls = controls
        self.num_controls = len(controls)

        # Check validity of starting state
        self._check_state(start_state)
        self.wind_measures_dim = start_state["wind_measurements"].shape[0]
        self.start_state = self._cast_dict_array(start_state)

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

        # Setup state space
        state_space_dict = {}
        for attr in self.STATE_ATTRIBUTES:
            if attr == "wind_measurements":
                low = np.array([3, 0], dtype=np.float32)
                high = np.array([28, 360], dtype=np.float32)
            elif attr in controls:
                low = np.ones(num_turbines, dtype=np.float32) * controls[attr][0]
                high = np.ones(num_turbines, dtype=np.float32) * controls[attr][1]
            else:
                low = start_state[attr].astype(np.float32)
                high = low
            state_space_dict[attr] = spaces.Box(
                low,
                high,
                shape=low.shape,
            )
        self.state_space = spaces.Dict(state_space_dict)

        # Take a first step in the interface
        self.start_state, _ = self.step_interface(self.start_state)

    @staticmethod
    def generate_start_state(num_turbines, start_actuators, start_wind):
        state = np.tile(np.c_[[start_wind], [start_actuators]], (num_turbines, 1))
        return state

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
            # if (len_b == 3) and self.continuous_control:
            #     raise Warning(
            #         f"A step size was provided for actuator {name} but `continuous_control`"
            #         " is activated. The step size will be ignored."
            #     )
            # if (len_b == 2) and not self.continuous_control:
            if len_b == 2:
                control_dict[name] = bounds_and_step + [1]
                raise Warning(
                    f"No step size was provided for actuator {name}. Step size will default to 1."
                )
                # raise Warning(
                #     f"No step size was provided for actuator {name} but `continuous_control`"
                #     " is not activated. Step size will default to 1."
                # )
            if not self.continuous_control:
                if (len_b == 3) and bounds_and_step[2] <= 0:
                    raise ValueError(
                        f"Invalid step size provided for actuator {name}"
                        " the step size must be stricly positive"
                    )
                if len_b == 2:
                    control_dict[name] = bounds_and_step + [1]
                    raise Warning(
                        f"No step size was provided for actuator {name} but `continuous_control`"
                        " is not activated. Step size will default to 1."
                    )

    def _check_matrix_state(self, state: np.ndarray):
        # Global state is a matrix of size NUM_TURBINES x (NUM_CONTROLS + WIND_MEASURES_DIM)
        expected_shape = (self.num_turbines, self.num_controls + self.wind_measures_dim)
        if state.shape != expected_shape:
            raise TypeError(
                f"Expected array of shape {expected_shape}, but got {state.shape}."
                " Global state should be an array of shape"
                " NUM_TURBINES x (NUM_CONTROLS + WIND_MEASURES_DIM)"
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
        self.interface.update_command(
            yaw=state["yaw"],
            pitch=state["pitch"],
            torque=state["torque"],
        )
        powers = self.interface.get_turbine_powers()
        state["wind_measurements"] = self.interface.get_turbine_wind()
        return state, powers / 1e6

    def take_action(self, state: Dict, joint_action: Dict):
        next_state = self.get_state_transition(state, joint_action)
        next_state, powers = self.step_interface(next_state)
        return next_state, powers

    def get_state_transition(self, state: Dict, joint_action: Dict):
        # Deterministic transition
        if not isinstance(joint_action, dict):
            raise TypeError("Joint action must be a dictionary")
        state = self._cast_dict_array(state)
        assert self.state_space.contains(state)
        joint_action = self._cast_dict_array(joint_action)
        next_state = copy.deepcopy(state)
        for control, command_joint_action in joint_action.items():
            assert control in self.controls, f"Control òf `{control}` is not activated"
            # if control not in state:
            #     raise ValueError(f"Received action for unknown command {control}")
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
