# import itertools
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from my_env.handlers.handler import Handler


class MComCentralHandler(Handler):
    @classmethod
    def env_obs_size(cls, env) -> int:
        return env.NUM_BS * 2 + env.NUM_UE * 2 + env.NUM_UAV * 3    # bs, ue, uav 位置

    @classmethod
    def uav_obs_size(cls, env) -> int:   # 用于定义observation_space，每个UAV观测的大小
        feature_sizes = env.feature_sizes
        return sum(feature_sizes[ftr] for ftr in feature_sizes)

    @classmethod
    def action_space(cls, env) -> spaces.MultiDiscrete:
        # define multi-discrete action space for central setting
        # each element of a multi-discrete action denotes one UE's decision
        # 每个UAV三维运动，有27种动作, [-1, 0, 1] * [-1, 0, 1] * [-1, 0, 1] 笛卡尔积
        
        return spaces.MultiDiscrete([len(env.uav_action_lst) for _ in env.UAV])

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        # observation is a single vector of concatenated UE representations
        # size = env.NUM_UAV * cls.uav_obs_size(env) + cls.env_obs_size(env)  # 各个UAV的观测 + 场景观测
        size = 876
        return spaces.Box(low=0.0, high=1.0, shape=(size,))

    @classmethod
    def action(cls, env, actions: Tuple[int]) -> Dict[int, Tuple[int]]:
        """Transform flattend actions to expected shape of core environment."""
        assert len(actions) == len(
            env.UAV
        ), "Number of actions must equal overall UAVs."

        uavs = sorted(env.UAV)
        return {uav_id: env.uav_action_lst[action] for uav_id, action in zip(uavs, actions)} # 每个UAVid : 对应动作(-1, 0, 1)

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Select from & flatten observations from MA setting."""
        # select observations considered in the central setting
        features = env.myLink.ObsDict
        start = True
        for i in features.values(): # 拉平拼接
            if start:
                arr = i.flatten()
                start = False
                continue
            arr = np.concatenate((arr, i.flatten()))
        return arr.astype(np.float32)

    @classmethod
    def reward(cls, env):
        """The central agent receives the average UE utility as reward."""
        # utilities = env.myLink.ObsDict['UE_MaxDR']
        return np.sum(env.myLink.ObsDict['UE_MaxDR'])   # 平均效用

    @classmethod
    def check(cls, env):
        pass
