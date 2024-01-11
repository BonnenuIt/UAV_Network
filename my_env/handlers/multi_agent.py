import copy
from typing import Dict

import gym
import numpy as np
from scipy.spatial.distance import pdist, squareform
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
    def action_space(cls, env) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                uav.uav_type + str(uav.uav_id): gym.spaces.Discrete(len(env.uav_action_lst))
                for uav in env.UAV.values()
            }
        )

    @classmethod
    def observation_space(cls, env) -> gym.spaces.Dict:
        # size = 66
        size = 120
        space = {
            uav.uav_type + str(uav.uav_id): gym.spaces.Box(low=0, high=1, shape=(size,), dtype=np.float64)
            for uav in env.UAV.values()
        }

        return gym.spaces.Dict(space)
    
    @classmethod
    def state_space(cls, env) -> gym.spaces.Dict:
        # size = 876
        size = 2100
        return gym.spaces.Box(low=0, high=1, shape=(size,), dtype=np.float64)

    @classmethod
    def reward(cls, myToolConn, DistUAV_UAV, agents, uav_team_id, curriculum_eplen):
        UE_DR = myToolConn.ObsDict['UE_MaxDR']
        NUM_UE_CONN = np.count_nonzero(UE_DR > 1e-6)   # np.count_nonzero（）获得True的数量
        UE_DR_TOTAL = np.sum(UE_DR)

        # uav_teams = np.zeros((len(agents)))
        # for i in range(len(uav_team_id)):
        #     uav_teams[:uav_team_id[len(uav_team_id) - i - 1]] = uav_team_id[len(uav_team_id) - i - 1]

        uav_conn_bs = np.where(np.array(myToolConn.Link_BS) >= 0, 1, 0) # 直连BS
        uav_conn_bs_factor = np.zeros((len(agents)))
        uav_conn_bs_factor[:uav_team_id[1]] = 0.1  # group 2 为 0.1
        uav_conn_bs_factor[:uav_team_id[0]] = 1     # group 1 为 1
        uav_conn_bs = uav_conn_bs * uav_conn_bs_factor

        uav_dist_punish = np.where(DistUAV_UAV < 0.3, -1, 0)
        row, col = np.diag_indices_from(uav_dist_punish)
        uav_dist_punish[row, col] = 0   # 去除自己对自己的零距离
        uav_dist_punish = np.sum(uav_dist_punish, axis=0)   # 距离惩罚

        relayReward = np.zeros((DistUAV_UAV.shape[0]))
        relayReward_01 = np.zeros((DistUAV_UAV.shape[0]))
        ue_L = np.where(UE_DR >= 1e-14)[0]  # 连接上的ue id
        UE_UAVid = np.array(myToolConn.UE_UAVid)[ue_L]  # 对应的uav id
        for i in range(ue_L.shape[0]):
            ue_i = ue_L[i]  # ue id
            UAV_i = UE_UAVid[i]   # 直连uav
            assert UAV_i != -1, "UAV_i == -1"
            allUAV = myToolConn.Link_backward[UAV_i] + [UAV_i] # 直连uav+中继 id
            for j in allUAV:
                relayReward[j] += UE_DR[ue_i]
                relayReward_01[j] += 1
        relay_factor = np.zeros((len(agents)))
        relay_factor[:uav_team_id[3]] = 0.2 # group 4
        relay_factor[:uav_team_id[2]] = 0.5 # group 3
        relay_factor[:uav_team_id[1]] = 1   # group 1, 2 为 1
        relayReward = relayReward * relay_factor
        relayReward_01 = relayReward_01 * relay_factor

        uav_bs_punish = np.zeros((DistUAV_UAV.shape[0]))
        Direct_BS_UAVid = np.where(np.array(myToolConn.Link_BS) > -1)[0]  # 直连BS的UAV
        for i in Direct_BS_UAVid:
            if len(myToolConn.Link_forward[i]) == 0:    # 无转发链路，只直连BS
                uav_bs_punish[i] = -1
        # uav_bs_punish[:2] = 0 # 考虑group 1不加惩罚
        uav_bs_punish_factor = np.zeros((len(agents)))
        uav_bs_punish_factor[:uav_team_id[3]] = 1   # group 4 为 1
        uav_bs_punish_factor[:uav_team_id[2]] = 0.9 # group 3 为 0.8
        uav_bs_punish_factor[:uav_team_id[1]] = 0.8 # group 2 为 0.5
        uav_bs_punish_factor[:uav_team_id[0]] = 0.7 # group 1 为 0.1
        uav_bs_punish = uav_bs_punish * uav_bs_punish_factor
        
        uav_unconn_punish = np.zeros((DistUAV_UAV.shape[0]))
        for i in range(len(myToolConn.Link_backward)):      # 无回程链路
            if isinstance(myToolConn.Link_backward[i], int):
                uav_unconn_punish[i] = -1

        Locs_UAV_xy = myToolConn.ObsDict['Locs_UAV'][:, :2] * myToolConn.normFactor
        Locs_UAV_xy = np.where(Locs_UAV_xy < 200, -1, Locs_UAV_xy)
        Locs_UAV_xy = np.where(Locs_UAV_xy > 2800, -1, Locs_UAV_xy)
        uav_pos_punish = np.min(np.where(Locs_UAV_xy == -1, -1, 0), axis=1)

        curriculum_punish = np.zeros((DistUAV_UAV.shape[0]))
        curriculum_totalLen = 20000000
        if curriculum_eplen < curriculum_totalLen:
            G1_Locs = np.array([[500, 500], [500, 2500], [2500, 500]])
            G2_Locs = np.array([[750, 750], [1250, 750], [750, 1250], [1250, 1250]])
            G3_Locs = np.array([[1750, 750], [1750, 1250], [1750, 1750], [750, 1750], [1250, 1750]])
            G4_Locs = np.array([[2250, 750], [2250, 1250], [2250, 1750], [2250, 2250], 
                                [750, 2250], [1250, 2250], [1750, 2250]])
            G_Locs = [G1_Locs, G2_Locs, G3_Locs, G4_Locs]
            Locs_UAV_xy_1 = myToolConn.ObsDict['Locs_UAV'][:, :2] * myToolConn.normFactor
            
            curriculum_punish = []
            for j in range(len(G_Locs)):
                if j == 0:
                    dist_2d = np.concatenate((Locs_UAV_xy_1[:uav_team_id[j], :], G_Locs[j]))
                    dist_2d = squareform(pdist(dist_2d))
                    dist_2d = dist_2d[uav_team_id[0]:, :uav_team_id[0]]
                else:
                    dist_2d = np.concatenate((Locs_UAV_xy_1[uav_team_id[j-1]:uav_team_id[j], :], G_Locs[j]))
                    dist_2d = squareform(pdist(dist_2d))
                    dist_len = uav_team_id[j] - uav_team_id[j-1]
                    dist_2d = dist_2d[dist_len:, :dist_len]
                
                for i in range(dist_2d.shape[1]):   # 每个uav循环
                    dist_i = np.argmin(dist_2d[:, i])
                    curriculum_punish.append(-dist_2d[dist_i, i]/1000)  # km
                    dist_2d = np.delete(dist_2d, dist_i, 0)
            curriculum_punish = np.array(curriculum_punish)

            if curriculum_eplen > (curriculum_totalLen / 2):
                curriculum_punish = curriculum_punish * (curriculum_totalLen - curriculum_eplen) / (curriculum_totalLen / 2)

        # if curriculum_eplen * 400 < 

        # UAV直连UE数目
        uav_ue_conn = np.where(myToolConn.myTools.O_DataRateUE_UAV > 0.001, 1, 0)
        uav_ue_conn = np.sum(uav_ue_conn, axis=0)
        uav_ue_conn_factor = np.zeros((len(agents)))
        uav_ue_conn_factor[:uav_team_id[3]] = 1     # group 4
        uav_ue_conn_factor[:uav_team_id[2]] = 0.9   # group 3
        uav_ue_conn_factor[:uav_team_id[1]] = 0.6   # group 2
        uav_ue_conn_factor[:uav_team_id[0]] = 0.1   # group 1
        uav_ue_conn = uav_ue_conn * uav_ue_conn_factor

        reward_factor = [0.1, 0.05, 0.05, 1, 0.15, 0.2, 0.1, 0.2, 0.1, 0.25, 8]
        all_rewards = NUM_UE_CONN     * reward_factor[0] + UE_DR_TOTAL       * reward_factor[1] \
                    + relayReward     * reward_factor[2] + uav_conn_bs       * reward_factor[3] \
                    + uav_dist_punish * reward_factor[4] + relayReward_01    * reward_factor[5] \
                    + uav_bs_punish   * reward_factor[6] + uav_unconn_punish * reward_factor[7] \
                    + uav_pos_punish  * reward_factor[8] + uav_ue_conn       * reward_factor[9] \
                    + curriculum_punish * reward_factor[10]
        
        ret = {}
        for i in range(len(agents)):
            ret[agents[i]] = all_rewards[i]

        all_reward_render = [NUM_UE_CONN, UE_DR_TOTAL, np.mean(relayReward), 
                             np.mean(uav_conn_bs), np.mean(uav_dist_punish), 
                             np.mean(relayReward_01), np.mean(uav_bs_punish)]

        return ret, NUM_UE_CONN, UE_DR_TOTAL, all_reward_render




    @classmethod
    def reward_old(cls, env):
        '''
            总
            1. 对每个用户提供的链接质量, 计算所有链接的通信速率 `UE_DR`, 并求和
            1. 连接上的用户数目 `NUM_UE_CONN`   # 解决覆盖性问题
            分
            1. 每个用户直连无人机和数据流经的中继无人机, 都加一个中继奖励, 即提供给此用户的通信速率
            1. 分组情况下, 离基站近的无人机直连上基站的数目  `uav_conn_bs`
            先用矩阵计算, 最后赋值给reward字典
        '''
        UE_DR = env.myLink.ObsDict['UE_MaxDR']
        DistUAV_UAV = env.myLink.DistUAV_UAV
        NUM_UE_CONN = np.count_nonzero(UE_DR > 1e-6)   # np.count_nonzero（）获得True的数量
        UE_DR_TOTAL = np.sum(UE_DR)

        uav_conn_bs = np.zeros((len(env.UAV)))
        uav_dist_punish = np.zeros((len(env.UAV)))
        relayReward = np.zeros((len(env.UAV)))
        relayReward_01 = np.zeros((len(env.UAV)))

        for uav in env.UAV.values():
            if uav.link2BS != None and len(uav.link2BS) == 1 and uav.uav_type == 'uavt1_':   # 和BS直连的UAV
                uav_conn_bs[uav.uav_id] = 1
            for i in range(DistUAV_UAV.shape[0]):   # UAV 间距惩罚
                if DistUAV_UAV[uav.uav_id, i] < 0.25 and uav.uav_id != i:
                    uav_dist_punish[uav.uav_id] += -1

        for ue_i in range(UE_DR.shape[0]):
            if UE_DR[ue_i] > 0:
                ue_uav_i = env.UE[ue_i].link2UAV[0] # link2UAV只有直连的UAV序号
                ue_uavs = env.UAV[env.agents[ue_uav_i]].link2BS[1:] + [ue_uav_i]    # 所有连接的UAV都加奖励
                for uav_i in ue_uavs:   # ue_uavs=[uav_ids, uav_ids, ...]
                    relayReward[uav_i] += UE_DR[ue_i]
                    relayReward_01[uav_i] += 1

        reward_factor = [0.4, 1/5, 0.05, 0, 0.002, 0.05]
        all_rewards = NUM_UE_CONN * reward_factor[0] + UE_DR_TOTAL * reward_factor[1] \
                        + relayReward * reward_factor[2] + uav_conn_bs * reward_factor[3] \
                              + uav_dist_punish * reward_factor[4]+ relayReward_01 * reward_factor[5]
        
        ret = {}
        for uav in env.UAV.values():
            ret[uav.uav_type + str(uav.uav_id)] = all_rewards[uav.uav_id]

        # ret = {}
        # for uav in env.UAV.values():
        #     ret[uav.uav_type + str(uav.uav_id)] = 0

        #     ### 把BS连接的奖励给 group uavt1_
        #     if uav.uav_type == 'uavt1_' and uav.link2BS != None:
        #         if len(uav.link2BS) == 1:
        #             ret[uav.uav_type + str(uav.uav_id)] = 0.03

        # # 中继的奖励
        # for ue_i in range(UE_DR.shape[0]):
        #     if UE_DR[ue_i] > 0:
        #         ue_uav_i = env.UE[ue_i].link2UAV[0] # link2UAV只有直连的UAV序号
        #         ue_uavs = env.UAV[env.agents[ue_uav_i]].link2BS[1:] + [ue_uav_i]    # 所有连接的UAV都加奖励
        #         for uav_i in ue_uavs:
        #             ret[env.agents[uav_i]] += UE_DR[ue_i]
        # # return np.mean(env.myLink.ObsDict['UE_MaxDR'])   # 平均效用

        # # 总奖励
        # total_rew = np.sum(UE_DR)

        # # 总奖励乘系数加上去 再总/50
        # for i in ret:
        #     if i[:6] == 'uavt3_':
        #         ret[i] += (total_rew * 2)
        #     else:
        #         ret[i] += (total_rew * 1.5)
        #     ret[i] /= 30
        return ret, NUM_UE_CONN, UE_DR_TOTAL
    
    @classmethod
    def reward_fcoop(cls, env):
        """UE's reward is their utility and the avg. utility of nearby BSs."""
        rewards = np.sum(env.myLink.ObsDict['UE_MaxDR'])
        ret = {uav.uav_type + str(uav.uav_id): rewards for uav in env.UAV.values()}
        return ret   # 平均效用

    @classmethod
    def my_global_state(cls, features) -> Dict[int, np.ndarray]:
        # features = env.myLink.ObsDict
        start = True
        for i in features.values(): # 拉平拼接
            if start:
                arr = i.flatten()
                start = False
                continue
            arr = np.concatenate((arr, i.flatten()))
        return arr
    # .astype(np.float64)

    @classmethod
    def observation(cls, features, agents) -> Dict[int, np.ndarray]:
        """Select features for MA setting & flatten each UE's features."""
        ret = {}
        # features = env.myLink.ObsDict
        for i in range(len(agents)):
            # i = env.UAV[uav_id].uav_id
            C1 = features['ConnBS_UAV'][:, i]
            C2 = features['ConnUAV_UAV'][:, i]
            C3 = features['DataRateUE_UAV'][:, i]
            # Loc = np.array([env.UAV[uav_id].x, env.UAV[uav_id].y, env.UAV[uav_id].height])
            Loc = features['Locs_UAV'][i, :]
            C_TOTAL = np.concatenate((C1, C2, C3, Loc))
            ret[agents[i]] = C_TOTAL
            # .astype(np.float64)
        return ret

    @classmethod
    def observation_old(cls, env) -> Dict[int, np.ndarray]:
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
            ret[uav.uav_type + str(uav.uav_id)] = copy.deepcopy(arr)
            # .astype(np.float64)
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
            ret[i] = env.uav_action_lst[actions[i]]
        return ret # 每个UAVid : 对应动作(-1, 0, 1)

