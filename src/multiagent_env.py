import json
from sre_constants import error
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import gym
from gym import spaces
import numpy as np
from pathlib import Path
from pettingzoo import AECEnv
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from envs import BaseFarmEnv
from interface import BaseInterface
from qlearning_utils import less_than_180, is_within_bounds


class DecentralizedFarmEnv(BaseFarmEnv, AECEnv):

    metadata = {"render_modes": ["human"], "name": "DecentralizedFarmEnv"}


    def __init__(self, 
        config: Path,
        interface: BaseInterface,
        state_lb: np.ndarray,
        state_ub: np.ndarray,
        actions: list,
        yaw_init: int = 0,
        velocity: float = 8.0,
        delta_yaw: float = 1.0,
        reward_tol: float = 0.0005,
        continuous_control: bool = False,
        action_bounds: tuple = (-1,1),
        reward_bounds: tuple = (-1,1),
        yaw_bound: float = 40.0,
        sync_turbines: bool = True,
        interface_kwargs = None
    ):
        """
            sync_turbines: all turbines act in the environment at the same time !
        """
        super(DecentralizedFarmEnv, self).__init__(
            config, interface, yaw_init, velocity, delta_yaw, reward_tol, 
                reward_bounds, yaw_bound, interface_kwargs
        )
        
        self.possible_agents = ["turbine_" + str(r+1) for r in range(self.num_turbines)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._continuous_control = continuous_control
        self._state_lb = state_lb
        self._state_ub = state_ub
        self._actions = actions
        self._action_bounds = action_bounds
        self._action_history = []
        self._sync_turbines = sync_turbines

        if continuous_control:
            self._action_spaces = {
                agent: spaces.Box(low=np.array([action_bounds[0]]), 
                                  high=np.array([action_bounds[1]]),
                                  shape=(1,), dtype=np.float32)
                for agent in self.possible_agents
            }
        else:
            self._action_space_start = -1
            self._action_spaces = {
                agent: spaces.Discrete(len(actions))
                for agent in self.possible_agents
            }

        self._obs_shape = state_ub.shape[0]
        self._observation_spaces = {
            agent: spaces.Box(low=state_lb, high=state_ub,
                    shape=(self._obs_shape,), dtype=np.float32)  
            for agent in self.possible_agents
        }
        self._get_action_space = self._build_action_space_method()

        # initialize state
        self._state = np.zeros((self.num_turbines,self._obs_shape))
        self._update_state()

    # @functools.lru_cache(maxsize=None)
    @property
    def observation_space(self):
        return spaces.Box(low=self._state_lb, high=self._state_ub,
                    shape=(self._obs_shape,), dtype=np.float32)

    def _build_action_space_method(self):
        if self._continuous_control:
            def action_space_fn(agent):
                return spaces.Box(low=np.array([self._action_bounds[0]]), 
                                  high=np.array([self._action_bounds[1]]),
                                  shape=(1,), dtype=np.float32)
        else:
            def action_space_fn(agent):
                return spaces.Discrete(len(self._actions))     
        return action_space_fn

    # @functools.lru_cache(maxsize=None)
    @property
    def action_space(self):
        return self._get_action_space(None)

    def _update_state(self):
        yaws = self.yaws
        self._state[:,0] = yaws
        self._state[:, 1] = self.direction
        self._state[:, -1] = self.velocity

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return  self._state[self.agent_name_mapping[agent]].copy()

    def reset(self, seed=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
       
        self.reset_farm() 
        self._update_state()

        self.agents = self.possible_agents[:]
        self._num_steps = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.actions = {agent: None for agent in self.agents}
        self.observations = {agent:  self._state[self.agent_name_mapping[agent]].copy() for agent in self.agents}
        self.num_moves = 0

        self.productions = {agent:  self._turbine_prods[-1][idx] for idx, agent in enumerate(self.agents)}

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def get_violation_conter(self):
        return self._bound_violation_counter.copy()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        agent = self.agent_selection
        agent_id = self.agent_name_mapping[agent]

        low = self.action_space.low[0]
        high = self.action_space.high[0]
        action = np.clip(
            action, 
            low,
            high
        )

        # get relative bounds
        greedy_yaw = self.sim_interface.get_greedy_yaw()
        relative_lower_yaw = less_than_180(greedy_yaw - self._yaw_bound)
        relative_upper_yaw = less_than_180(greedy_yaw + self._yaw_bound)
        relative_bounds = [relative_lower_yaw, relative_upper_yaw]
        within_relative_bounds, distances = is_within_bounds(self.yaws[agent_id], relative_lower_yaw, relative_upper_yaw)

        # The old yaw is always within the absolute bounds, but it may not be within
        # the relative bounds as the wind keeps changing

        if not within_relative_bounds:
            green_yaw = relative_bounds[np.abs(distances).argmin()]
            self._bound_violation_counter[agent_id] += 1 

        self._num_steps[agent] += 1
        old_yaws = self.yaws
        new_yaws = old_yaws.copy()
        if self._continuous_control:
            new_yaw = old_yaws[agent_id] + action
        else:
            new_yaw = old_yaws[agent_id] + self._delta_yaw * (action + self._action_space_start)
        new_yaw = np.clip(
            new_yaw, 
            self.observation_space.low[0],
            self.observation_space.high[0]
        )
        new_yaws[agent_id] = new_yaw

        self.sim_interface.set_yaw_angles(new_yaws, sync=not self._sync_turbines)
        self._update_state()

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        # stores action of current agent
        self.actions[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            if self._sync_turbines:
                self.sim_interface.set_yaw_angles(self.yaws, sync=True)
            self._prods.append(self.sim_interface.get_farm_power())
            self._turbine_prods.append(self.evaluate_power())
            self.productions = {agent:  self._turbine_prods[-1][idx] for idx, agent in enumerate(self.agents)}
            self._yaws.append(new_yaws)
            self._action_history.append(list(self.actions.values()))
            reward = np.array(self._get_reward())
            for agent in self.agents:
                # same reward for everybody
                self.rewards[agent] = reward
                self.observations[agent] = self._state[self.agent_name_mapping[agent]].copy()
                # self.infos[agent] = {"greedy_yaw": greedy_yaw}
            self.sim_interface.next_wind()
            self.num_moves += 1
        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()