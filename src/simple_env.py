import copy
from typing import Dict

import gymnasium as gym
import numpy as np

from src.interface import BaseInterface
from src.mdp import WindFarmMDP
from src.rewards import RewardShaper


class WindFarmEnv(gym.Env):
    def __init__(
        self,
        interface: BaseInterface,
        num_turbines: int,
        controls: dict,
        continuous_control: bool = True,
        interface_kwargs: Dict = None,
        reward_shaper: RewardShaper = None,
    ):
        # TODO: make start_state
        start_state = {
            "wind_measurements": np.array([8, 0]),
            "yaw": np.array([0, 0, 0]),
            "pitch": np.array([0, 0, 0]),
            "torque": np.array([0, 0, 0]),
        }

        self.mdp = WindFarmMDP(
            interface=interface,
            num_turbines=num_turbines,
            controls=controls,
            start_state=start_state,
            continuous_control=continuous_control,
            interface_kwargs=interface_kwargs,
        )
        self.continuous_control = continuous_control
        self.action_space = self.mdp.action_space
        self.observation_space = self.mdp.state_space
        self._state = self.mdp.start_state
        self.num_turbines = self.mdp.num_turbines
        if reward_shaper is not None:
            self.reward_shaper = reward_shaper
        else:
            self.reward_shaper = lambda x: x

    def reset(self):
        pass

    def step(self, actions: Dict):
        """
        action: np.array of shape (n_turbines,)
        """
        next_state, powers = self.mdp.take_action(self._state, actions)
        reward = np.array([self.reward_shaper(powers.sum())])
        self._state = next_state
        terminated = np.array(False)
        truncated = np.array(False)
        info = {"power": powers}
        observation = copy.deepcopy(self._state)
        return observation, reward, terminated, truncated, info
