import functools
from typing import Dict

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from src.interface import BaseInterface
from src.mdp import WindFarmMDP
from src.rewards import RewardShaper


class MAWindFarmEnv(AECEnv):
    metadata = {"name": "multiagent-windfarm"}

    def __init__(
        self,
        interface: BaseInterface,
        num_turbines: int,
        controls: dict,
        continuous_control: bool = True,
        interface_kwargs: Dict = None,
        reward_shaper: RewardShaper = None,
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
        self.max_num_steps = max_num_steps
        self._state = self.mdp.start_state
        self.num_turbines = self.mdp.num_turbines
        if reward_shaper is not None:
            self.reward_shaper = reward_shaper
        else:
            self.reward_shaper = lambda x: x

        # Init AEC properties
        self.possible_agents = [
            "turbine_" + str(r + 1) for r in range(self.num_turbines)
        ]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._build_agent_spaces()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def state(self):
        return self._state

    def _build_agent_spaces(self):
        # Retrieve observation and action spaces for all agents
        self._obs_spaces = {}
        self._action_spaces = {}
        for i, agent in enumerate(self.possible_agents):
            self._obs_spaces[agent] = {
                key: (
                    space
                    if key == "wind_measurements"
                    else spaces.Box(space.low[i], space.high[i])
                )
                for key, space in self.mdp.state_space.items()
            }
            if self.continuous_control:
                self._action_spaces[agent] = {
                    key: spaces.Box(space.low[i], space.high[i])
                    for key, space in self.mdp.action_space.items()
                }
            else:
                self._action_spaces[agent] = {
                    key: space[i] for key, space in self.mdp.action_space.items()
                }

    def _join_actions(self, agent_actions):
        joint_action = {
            control: np.zeros(self.num_turbines, dtype=np.float32)
            for control in self.mdp.controls
        }
        for j, (agent, action) in enumerate(agent_actions.items()):
            for control in action:
                joint_action[control][j] = action[control][:]
        # TODO: add proper handling of logging
        # (debug level) print(f"Created joint action {joint_action}")
        return joint_action

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        global_state = self.state()
        agent_state = {}
        agent_state["wind_measurements"] = global_state["wind_measurements"]
        for key, partial_state in global_state.items():
            if key != "wind_measurements":
                agent_state[key] = partial_state[self.agent_name_mapping[agent]]
        return agent_state

    def reset(self, seed=None):
        """
        Reset initializes the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """

        self.agents = self.possible_agents[:]
        self._num_steps = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.actions = {agent: None for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0
        """
        Init agent selector
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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
        self._num_steps[agent] += 1

        # TODO: allow for different control for each agent
        # For now, every local action must send a command for ALL controls
        if any([not (control in action) for control in self.mdp.controls]):
            raise ValueError(
                f"Action {action} for agent {agent} is incomplete."
                f" List of needed controls: {self.mdp.controls.keys()}"
            )

        # restart reward accumulation
        self._cumulative_rewards[agent] = 0
        # stores action of current agent
        self.actions[self.agent_selection] = action

        # collect reward when all agents have taken an action
        if self._agent_selector.is_last():
            next_state, powers, loads, truncated = self.mdp.take_action(
                self._state, self._join_actions(self.actions)
            )
            reward = np.array([self.reward_shaper(powers.sum())])
            self._state = next_state
            for agent in self.agents:
                # cooperative env: same reward for everybody
                # might change later to account for fatigue
                self.rewards[agent] = reward
                self.observations[agent] = self.observe(agent)
                self.truncations[agent] = truncated
                self.terminations[agent] = False
                self.infos[agent] = {
                    "power": powers[self.agent_name_mapping[agent]],
                }
                if loads is not None:
                    self.infos[agent]["load"] = loads[self.agent_name_mapping[agent]]
            if truncated:
                self.agents = []
            self.num_moves += 1
        else:
            # no reward allocated until all players take an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
