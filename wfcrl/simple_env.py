import copy
from typing import Dict

import gymnasium as gym
import numpy as np

from wfcrl.environments import FarmCase
from wfcrl.interface import BaseInterface
from wfcrl.mdp import WindFarmMDP
from wfcrl.rewards import DoNothingReward, RewardShaper


class WindFarmEnv(gym.Env):
    metadata = {"name": "centralized-windfarm"}

    def __init__(
        self,
        interface: BaseInterface,
        farm_case: FarmCase,
        controls: dict,
        continuous_control: bool = True,
        reward_shaper: RewardShaper = DoNothingReward(),
        start_iter: int = 0,
        max_num_steps: int = 500,
        load_coef: float = 0.1
    ):
        self.mdp = WindFarmMDP(
            interface=interface,
            farm_case=farm_case,
            controls=controls,
            continuous_control=continuous_control,
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
        self.dt = farm_case.dt
        self.farm_case = farm_case
        self.accumulated_actions = self.mdp.get_accumulated_actions()
        self.num_moves = 0
        self.load_coef = load_coef

    def reset(self, seed=None, options=None):
        self.mdp.reset(seed, options)
        self._state = self.mdp.start_state
        self.reward_shaper.reset()
        observation = copy.deepcopy(self._state)
        self.accumulated_actions = self.mdp.get_accumulated_actions()
        self.num_moves = 0
        return observation

    def step(self, actions: Dict):
        """
        action: dictionary of np.array of shape (num_turbines,)
        """
        assert self._state is not None, "Call reset before `step`"

        self.num_moves += 1
        for control in actions:
            if not (control in self.mdp.ACTUATORS_RATE):
                continue
            actuating_time = (
                self.accumulated_actions[control] / self.mdp.ACTUATORS_RATE[control]
            )
            actuating_frac = actuating_time / self.num_moves / self.farm_case.dt
            actions[control][actuating_frac >= 0.1] = 0.0

        next_state, powers, loads, truncated = self.mdp.take_action(
            self._state, actions
        )
        # normalize by initial freestream wind
        normalized_powers = (
            powers * 1e3 / (self._state["freewind_measurements"][0] ** 3)
        )
        load_penalty = 0
        if loads is not None:
            load_penalty = np.mean(np.abs(loads))
        reward = normalized_powers.mean() - self.load_coef * load_penalty
        reward = np.array([self.reward_shaper(reward)])
        self._state = next_state
        terminated = False
        truncated = truncated
        info = {"power": powers}
        if loads is not None:
            info["load"] = loads
        observation = copy.deepcopy(self._state)

        # accumulate action for constraint checking
        self.accumulated_actions = self.mdp.get_accumulated_actions()
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
