import numpy as np

from wfcrl.interface import MPI_Interface
from wfcrl.mdp import WindFarmMDP
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.rewards import StepPercentage
from wfcrl.simple_env import WindFarmEnv

interface_params = {
    "measurement_window": 10,
    "buffer_size": 50_000,
    "log_file": "log.txt",
}
controls = {"yaw": (-20, 20, 15), "pitch": (0, 45, 1)}  # "torque": (0, 2e4, 5e3)}
start_state = {
    "wind_measurements": np.array([8, 0]),
    "yaw": np.array([0, 0, 0]),
    "pitch": np.array([0, 0, 0]),
    "torque": np.array([0, 0, 0]),
}
joint_action = {"yaw": np.array([0, 0, 0]), "pitch": np.array([0, 0, 0])}
# local_action = {"yaw": np.array([0])}
local_action = {"yaw": np.array([0]), "pitch": np.array([0])}


def test_mdp(mdp, T=20):
    state = mdp.start_state
    for i in range(T):
        joint_action["yaw"][0] = 0
        if i == 20:
            joint_action["yaw"][0] = 30
        state, powers, loads, done = mdp.take_action(state, joint_action)
        print(f"Step {i}: powers {powers}")
    return


def gym_routine(env, T=20):
    """Gymnasium RL routine"""
    env.reset()
    r = 0
    for i in range(T):
        joint_action["yaw"][0] = 0
        if i == 20:
            env.step(
                {
                    "yaw": np.array([15, 0, 0]),
                    "pitch": np.array([1, 1, 2]),
                    "torque": np.array([4e3, 4e3, 7e3]),
                }
            )
            # joint_action["yaw"][0] = 30
        observation, reward, done, info = env.step(joint_action)
        r += reward
        print(f"reward = {r}")
    print(f"Total reward = {r}")
    return


def pz_routine(env, T=20):
    """Petting Zoo MARL Routine"""
    env.reset()
    r = {agent: 0 for agent in env.possible_agents}
    i = 0
    totalT = env.num_turbines * T
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        r[agent] += reward
        i += 1
        if i > totalT:
            break
        if i == 20:
            env.step(
                {
                    "yaw": np.array([15.0]),
                    "pitch": np.array([3.0]),
                    # "torque": np.array([4e3]),
                }
            )
        env.step(local_action)

        print(f"reward = {r}")
    print(f"Total reward = {r}")


if __name__ == "__main__":
    test = "pettingzoo"  # ["mdp", "gym" "pettingzoo"]
    if test == "mdp":
        mdp = WindFarmMDP(
            interface=MPI_Interface,
            num_turbines=3,
            controls=controls,
            interface_kwargs=interface_params,
        )
        test_mdp(mdp, start_state, T=100)
    elif test == "gym":
        env = WindFarmEnv(
            interface=MPI_Interface,
            num_turbines=3,
            controls=controls,
            interface_kwargs=interface_params,
            reward_shaper=StepPercentage(),
        )
        gym_routine(env, T=40)
    elif test == "pettingzoo":
        env = MAWindFarmEnv(
            interface=MPI_Interface,
            num_turbines=3,
            controls=controls,
            interface_kwargs=interface_params,
            reward_shaper=None,
        )
        pz_routine(env, T=40)
