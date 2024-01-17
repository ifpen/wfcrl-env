from numbers import Number
import time
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import gym
from gym import spaces
import numpy as np
from pathlib import Path

from interface import BaseInterface


class BaseFarmEnv(gym.Env):

    optimal_conditions = {
        "max_power": 4e6
    }
    def __init__(self, 
        config: Path, 
        interface: BaseInterface,
        yaw_init: Union[Number, List] = 0,
        velocity: float = 8.0,
        delta_yaw: float = 1.0,
        reward_tol: float = 0.0005,
        reward_bounds: Tuple = (-1,1),
        yaw_bound : float = 40.0,
        interface_kwargs = None
    ):

        super().__init__()
        
        self.config = config
        self._delta_yaw = delta_yaw
        self._reward_tol = reward_tol
        self._reward_abs_values = np.arange(0,reward_bounds[1]+1,1)
        self._yaw_bound = yaw_bound
        
        # Init FLORIS (or other) interface
        if interface_kwargs is None:
            interface_kwargs = {}
        self.sim_interface = interface(config, **interface_kwargs)
        self.num_turbines = self.sim_interface.num_turbines
        self._bound_violation_counter = np.zeros(self.num_turbines)
        if isinstance(yaw_init, Number):
            self._reset_yaws = [yaw_init for _ in range(self.num_turbines)]
        else:
            assert len(yaw_init) == self.num_turbines
            self._reset_yaws = yaw_init
        
        self.sim_interface.reset_interface(
            yaws=self._reset_yaws, 
            wind_direction=270.0,
            wind_speed=velocity,
            wind_shear=0.,
            turbulence_intensity=0.08
        )
        #Init tracking lists
        self._prods = []
        self._turbine_prods = []
        self._yaws = []
        self._num_resets = 0
        self._num_steps = 0
        self._resets = []
        self._percentages = []
        self._rewards = []
        self._times = []


    @property
    def velocity(self):
        return self.sim_interface.wind_speed
    
    @property
    def direction(self):
        return self.sim_interface.wind_dir

    @property
    def yaws(self):
        return np.array(
            self.sim_interface.get_yaw_angles(),
            dtype=np.float32
        )

    def _get_reward(self):
        if len(self._prods) < 1:
            raise Exception("`reset` environment before any step")
        reward = 0
        percentage = (self._prods[-1] - self._prods[-2]) / self._prods[-2]
        self._percentages.append(percentage)
        if np.abs(percentage) > self._reward_tol:
            magnitude = self._reward_abs_values[np.sum(np.abs(percentage) > self._reward_abs_values*self._reward_tol)-1]
            reward = np.sign(percentage) * magnitude
        self._rewards.append(reward)
        return reward

    def _get_novel_reward(self):
        if len(self._prods) < 1:
            raise Exception("`reset` environment before any step")
        reward = 0
        # reward = (self._prods[-1] - self.optimal_conditions["max_power"]) / 1e6
        # hyp_std = self._prods[-2] * 0.08

        percentage = (self._prods[-1] - self._prods[-2]) / self._prods[-2]
        self._percentages.append(percentage)
        if np.abs(percentage) > self._reward_tol:
            reward = percentage * 1000
        self._rewards.append(reward)
        return reward

    def evaluate_power(self):
        return self.sim_interface.get_turbine_powers()

    def reset_farm(self, reset_interface_args=None):
        self._num_resets += 1
        if reset_interface_args is None:
            reset_interface_args = {"yaws":  self._reset_yaws}
        self.sim_interface.reset_interface(**reset_interface_args)
        self._turbine_prods= [self.sim_interface.get_turbine_powers()]
        self._prods = [self.sim_interface.get_farm_power()]
        self._yaws = [self.yaws]
        self._times = [time.time()]

    def render(self, mode='human'):
        pass

    def close (self):
        return

    def reset(self):
        return NotImplemented

class FarmEnv(BaseFarmEnv):
    def __init__(self, 
        config: Path, 
        interface: BaseInterface,
        state_lb: np.ndarray,
        state_ub: np.ndarray,
        yaw_init: int = 0,
        velocity: float = 8.0,
        delta_yaw: float = 1.0,
        reward_tol: float = 0.0005,
        continuous_control: bool = False,
        actions: Iterable = np.array([-1,0,1]),
        action_bounds: tuple = (-1,1),
        reward_bounds: tuple = (-1,1),
        yaw_bound : float = 40.0,
        interface_kwargs = None,
    ):
        """
        config: path to floris parameters (.json file)
        lb, ub: bound on state space
        action_bounds: only for continuous action space
        """
        super(FarmEnv, self).__init__(
            config, interface, yaw_init, velocity, delta_yaw, reward_tol,
            reward_bounds, yaw_bound, interface_kwargs
        )

        self._state = np.zeros_like(state_lb, dtype=np.float32)
        self._update_state()
        self._continuous_control = continuous_control

        # continuous
        if continuous_control:
            self.action_space = spaces.Box(low=np.ones(self.num_turbines)*action_bounds[0], 
                                            high=np.ones(self.num_turbines)*action_bounds[1],
                                            shape=(self.num_turbines,), dtype=np.float32)
        else:
            self._action_space_start = -1
            # discrete
            self.action_space = spaces.MultiDiscrete(
                [len(actions) for _ in range(self.num_turbines)]
            )

        self.observation_space = spaces.Box(low=state_lb, high=state_ub,
                                        shape=state_lb.shape, dtype=np.float32)

    def _update_state(self):
        yaws = self.yaws
        self._state[:yaws.shape[0]] = yaws
        self._state[yaws.shape[0]] = self.direction
        self._state[-1] = self.velocity

    def step(self, actions):
        """
        action: np.array of shape (n_turbines,)
        """
        self._num_steps += 1
        old_yaws = self.yaws
        if self._continuous_control:
            new_yaws = old_yaws + actions
        else:
            new_yaws = old_yaws + self._delta_yaw * (actions + self._action_space_start)
        greedy_yaw = self.sim_interface.get_greedy_yaw()
        upper_yaws = np.minimum(greedy_yaw + self._yaw_bound, self.observation_space.high[:self.num_turbines])
        lower_yaws = np.maximum(greedy_yaw - self._yaw_bound, self.observation_space.low[:self.num_turbines])
        new_yaws = np.clip(
            new_yaws, 
            self.observation_space.low[0],
            self.observation_space.high[0]
        )

        self.sim_interface.set_yaw_angles(yaws=new_yaws)
        self._update_state()
        observation = self._state.copy()
        done = np.array(False)
        info = {}
        self._turbine_prods.append(self.evaluate_power())
        self._prods.append(self.sim_interface.get_farm_power())
        if np.isnan(self._prods[-1]):
            self.sim_interface.get_farm_power()
            raise ValueError("Invalid Production.")
        self._times.append(time.time())
        self._yaws.append(new_yaws)
        reward = np.array(self._get_reward())
        return observation, reward, done, info

    def reset(self, start=None):
        reset_interface_args = {}
        if start is not None:
            reset_interface_args["yaws"] = start[:self.num_turbines]
            reset_interface_args["wind_direction"] = start[-2]
            reset_interface_args["wind_speed"] = start[-1]

        self.reset_farm(reset_interface_args)
        self._update_state()
        observation = self._state.copy()
        return  observation # reward, done, info can't be included