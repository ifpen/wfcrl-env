import numpy as np

from src.cases import farm_3_floris as case
from src.interface import FlorisInterface
from src.multiagent_env import MAWindFarmEnv
from src.rewards import StepPercentage

controls = {"yaw": (-20, 20, 15)}


def dummy_policy(agent, i):
    if (agent == "turbine_1") and (i == 20):
        return {
            "yaw": np.array([15.0]),
        }
    return {"yaw": np.array([0])}


env = MAWindFarmEnv(
    interface=FlorisInterface,
    num_turbines=case.n_turbines,
    controls=controls,
    interface_kwargs=case.interface_kwargs,
    reward_shaper=StepPercentage(),
    start_iter=int(np.ceil(case.t_init / case.dt)),
    max_num_steps=50,
)
env.reset()
r = {agent: 0 for agent in env.possible_agents}
done = {agent: False for agent in env.possible_agents}
num_steps = {agent: 0 for agent in env.possible_agents}
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    done[agent] = done[agent] or termination or truncation
    r[agent] += reward
    action = dummy_policy(agent, num_steps[agent])
    num_steps[agent] += 1
    env.step(action)

print(f"Total reward = {r}")
