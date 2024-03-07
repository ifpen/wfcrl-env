import numpy as np

from wfcrl import environments as envs
from wfcrl.rewards import StepPercentage

env = envs.make(
    "Dec_Ablaincourt_Fastfarm",
    max_num_steps=10,
    controls=["yaw", "pitch"],
    reward_shaper=StepPercentage(),
)


def dummy_policy(agent, i):
    if (agent == "turbine_1") and (i == 20):
        return {
            "yaw": np.array([15.0]),
            "pitch": np.array([3.0]),
        }
    return {"yaw": np.array([0]), "pitch": np.array([0.0])}


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
