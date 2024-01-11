import copy
from typing import Dict

import gymnasium
import numpy as np

from my_env.handlers.handler import Handler


class MComMAHandler(Handler):
    # features = [
    #     "connections",
    #     "snrs",
    #     "utility",
    #     "bcast",
    #     "stations_connected",
    # ]

    # @classmethod
    # def ue_obs_size(cls, env) -> int:
    #     return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict(
            {
                str(uav.uav_id): gymnasium.spaces.Discrete(len(env.uav_action_lst))
                for uav in env.UAV.values()
            }
        )

    @classmethod
    def observation_space(cls, env) -> gymnasium.spaces.Dict:
        size = 876
        space = {
            str(uav.uav_id): gymnasium.spaces.Box(low=-1, high=1, shape=(size,), dtype=np.float32)
            for uav in env.UAV.values()
        }

        return gymnasium.spaces.Dict(space)

    @classmethod
    def reward(cls, env):
        """UE's reward is their utility and the avg. utility of nearby BSs."""
        # # compute average utility of UEs for each BS
        # # set to lower bound if no UEs are connected
        # bs_utilities = env.station_utilities()

        # def ue_utility(ue):
        #     """Aggregates UE's own and nearby BSs' utility."""
        #     # ch eck what BS-UE connections are possible
        #     connectable = env.available_connections(ue)

        #     # utilities are broadcasted, i.e., aggregate utilities of BSs
        #     # that are in range of the UE
        #     ngbr_utility = sum(bs_utilities[bs] for bs in connectable)

        #     # calculate rewards as average weighted by
        #     # the number of each BSs' connections
        #     ngbr_counts = sum(len(env.connections[bs]) for bs in connectable)

        #     return (ngbr_utility + env.utilities[ue]) / (ngbr_counts + 1)

        # rewards = {ue.ue_id: ue_utility(ue) for ue in env.active}
        # return rewards
        rewards = env.myLink.ObsDict['UE_MaxDR']
        ret = {str(uav.uav_id): 0 for uav in env.UAV.values()}
        for ue_i in range(rewards.shape[0]):
            if rewards[ue_i] > 0:
                ue_uav_i = env.UE[ue_i].link2UAV[0]
                ue_uavs = env.UAV[ue_uav_i].link2BS + [ue_uav_i]
                for uav_i in ue_uavs:
                    ret[str(uav_i)] += rewards[ue_i]
        # return np.mean(env.myLink.ObsDict['UE_MaxDR'])   # 平均效用
        return ret

    @classmethod
    def observation(cls, env) -> Dict[int, np.ndarray]:
        """Select features for MA setting & flatten each UE's features."""
        ret = {}
        features_origin = env.myLink.ObsDict
        for uav in env.UAV.values():
            features = copy.deepcopy(features_origin)
            if uav.uav_id != 0: # 和0号uav交换行列
                features['ConnBS_UAV'][:, [0, uav.uav_id]] = features['ConnBS_UAV'][:, [uav.uav_id, 0]]
                # 交换列
                features['ConnUAV_UAV'][:, [0, uav.uav_id]] = features['ConnUAV_UAV'][:, [uav.uav_id, 0]]
                features['ConnUAV_UAV'][[0, uav.uav_id], :] = features['ConnUAV_UAV'][[uav.uav_id, 0], :]

                features['DataRateUE_UAV'][:, [0, uav.uav_id]] = features['DataRateUE_UAV'][:, [uav.uav_id, 0]]
                features['UAV_Available_Lst_onehot'][[0, uav.uav_id]] = features['UAV_Available_Lst_onehot'][[uav.uav_id, 0]]
                features['Locs_UAV'][[0, uav.uav_id], :] = features['Locs_UAV'][[uav.uav_id, 0], :]

                # self.ObsDict["UE_UAVid"] = (np.array(self.UE_UAVid)+1) / env.NUM_UAV        # 0 - #uav数目
                # id_0, id_thisUAV = -1, -1
                for i in range(features['UE_UAVid'].shape[0]):
                    if features['UE_UAVid'][i] == ((0 + 1) / env.NUM_UAV):
                        features['UE_UAVid'][i] = (uav.uav_id + 1) / env.NUM_UAV
                        continue
                    if features['UE_UAVid'][i] == ((uav.uav_id + 1) / env.NUM_UAV):
                        features['UE_UAVid'][i] = ((0 + 1) / env.NUM_UAV)
                        continue

            start = True
            for i in features.values(): # 拉平拼接
                if start:
                    arr = i.flatten()
                    start = False
                    continue
                arr = np.concatenate((arr, i.flatten()))
            ret[str(uav.uav_id)] = arr.astype(np.float32)
        return ret
        # # get features for currently active UEs
        # active = set([ue.ue_id for ue in env.active if not env.time_is_up])
        # features = env.features()
        # features = {ue_id: obs for ue_id, obs in features.items() if ue_id in active}

        # # select observations for multi-agent setting from base feature set
        # obs = {
        #     ue_id: [obs_dict[key] for key in cls.features]
        #     for ue_id, obs_dict in features.items()
        # }

        # # flatten each UE's Dict observation to vector representation
        # obs = {
        #     ue_id: np.concatenate([o for o in ue_obs]) for ue_id, ue_obs in obs.items()
        # }
        # return obs

    @classmethod
    def action(cls, env, actions: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        # return action
        assert len(actions) == env.NUM_UAV, "Number of actions must equal overall UAVs."

        uavs = sorted(env.UAV)
        ret = {}
        for i in uavs:
            ret[i] = env.uav_action_lst[actions[str(i)]]
        return ret # 每个UAVid : 对应动作(-1, 0, 1)

