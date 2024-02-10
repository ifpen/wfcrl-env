import copy
from collections import OrderedDict
from typing import Dict, Iterable, Type

import numpy as np
from gymnasium import spaces

from interface import BaseInterface
from reward import PercentageReward, WindFarmReward


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
        reward: Type[WindFarmReward] = PercentageReward,
        continuous_control: bool = True,
        interface_kwargs: dict = {},
    ):
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
        self.start_state = start_state

        # Setup actions
        if self.continuous_control:
            self.action_space = spaces.Dict(
                {
                    name: spaces.Box(
                        bounds_and_step[0],
                        bounds_and_step[1],
                        shape=(self.num_turbines,),
                    )
                    for name, bounds_and_step in self.controls
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    name: spaces.MultiDiscrete(
                        [-bounds_and_step[2], 0, bounds_and_step[2]],
                        shape=(self.num_turbines,),
                    )
                    for name, bounds_and_step in self.controls
                }
            )

    @staticmethod
    def generate_start_state(num_turbines, start_actuators, start_wind):
        state = np.tile(np.c_[[start_wind], [start_actuators]], (num_turbines, 1))
        return state

    def _check_controls(self, control_dict: Dict):
        for name, bounds_and_step in control_dict:
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
            if (len_b == 3) and self.continuous_control:
                raise Warning(
                    f"A step size was provided for actuator {name} but `continuous_control`"
                    " is activated. The step size will be ignored."
                )
            if (len_b == 2) and not self.continuous_control:
                control_dict[name] = bounds_and_step + [1]
                raise Warning(
                    f"No step size was provided for actuator {name} but `continuous_control`"
                    " is not activated. Step size will default to 1."
                )
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
        for attr, value in state:
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

    def _continuous_control_transition(self, state: Dict, joint_action: Dict):
        next_state = copy.deepcopy(state)
        for control, command_joint_action in joint_action:
            # if control not in state:
            #     raise ValueError(f"Received action for unknown command {control}")
            next_state[control] = state[control] + command_joint_action
        return next_state

    def get_state_transition(self, state: np.ndarray, joint_action: Dict):
        # Deterministic transition
        if isinstance(joint_action, dict):
            raise TypeError("Joint action must be a dictionary")
        return state
