import numpy as np

from src.cases import farm_3_fastfarm as case
from src.interface import MPI_Interface
from src.multiagent_env import MAWindFarmEnv
from src.rewards import StepPercentage

controls = {"yaw": (-20, 20, 15), "pitch": (0, 45, 1), "torque": (-2e4, 2e4, 5e3)}
T = 40


def dummy_policy(i):
    if i == 20:
        return {
            "yaw": np.array([15.0]),
            "pitch": np.array([3.0]),
            "torque": np.array([4e3]),
        }
    return {"yaw": np.array([0]), "pitch": np.array([0]), "torque": np.array([0])}


env = MAWindFarmEnv(
    interface=MPI_Interface,
    num_turbines=3,
    controls=controls,
    interface_kwargs=case.interface_kwargs,
    reward_shaper=StepPercentage(),
    start_iter=int(np.floor(case.t_init / case.dt)),
)
env.reset()
r = {agent: 0 for agent in env.possible_agents}
i = 0
totalT = env.num_turbines * T
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(agent, info)
    r[agent] += reward
    i += 1
    if i >= totalT:
        break
    action = dummy_policy(i)
    env.step(action)
print(f"Total reward = {r}")
