from gymnasium.envs.registration import register

register(
    id="3turbines-FF-v0",
    entry_point="src.multiagent_env:MAWindFarmEnv",
    max_episode_steps=50,
)

register(
    id="3turbines-floris-v0",
    entry_point="src.multiagent_env:MAWindFarmEnv",
    max_episode_steps=50,
)
