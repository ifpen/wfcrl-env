import copy
from typing import Dict

import gymnasium as gym
import numpy as np

from wfcrl.interface import BaseInterface
from wfcrl.mdp import WindFarmMDP
from wfcrl.rewards import DoNothingReward, RewardShaper


class WindFarmEnv(gym.Env):
    def __init__(
        self,
        interface: BaseInterface,
        num_turbines: int,
        controls: dict,
        continuous_control: bool = True,
        interface_kwargs: Dict = None,
        reward_shaper: RewardShaper = DoNothingReward(),
        start_iter: int = 0,
        max_num_steps: int = 500,
    ):
        self.mdp = WindFarmMDP(
            interface=interface,
            num_turbines=num_turbines,
            controls=controls,
            continuous_control=continuous_control,
            interface_kwargs=interface_kwargs,
            start_iter=start_iter,
            horizon=start_iter + max_num_steps,
        )
        self.continuous_control = continuous_control
        self.action_space = self.mdp.action_space
        self.observation_space = self.mdp.state_space
        self._state = self.mdp.start_state
        self.num_turbines = self.mdp.num_turbines
        self.max_num_steps = max_num_steps
        self.reward_shaper = reward_shaper
        self.controls = controls

    def reset(self, seed=None, options=None):
        self.mdp.reset()
        self._state = self.mdp.start_state
        self.reward_shaper.reset()
        observation = copy.deepcopy(self._state)
        return observation

    def step(self, actions: Dict):
        """
        action: dictionary of np.array of shape (n_turbines,)
        """
        assert self._state is not None, "Call reset before `step`"

        next_state, powers, loads, truncated = self.mdp.take_action(
            self._state, actions
        )
        reward = np.array([self.reward_shaper(powers.sum())])
        self._state = next_state
        terminated = False
        truncated = truncated
        info = {"power": powers}
        if loads is not None:
            info["load"] = loads
        observation = copy.deepcopy(self._state)
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
