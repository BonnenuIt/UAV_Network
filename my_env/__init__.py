try:
    import gymnasium

    gymnasium.envs.register(
        id=f"myCommEnv-central-v1",
        entry_point=f"my_env.base:myCommEnvCentral",
    )

    gymnasium.envs.register(
        id=f"myCommEnv-ma-str",
        entry_point=f"my_env.base_marl_str:myCommEnvMA",
    )
except:
    pass