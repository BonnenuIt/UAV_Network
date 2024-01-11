import copy
import itertools
import sys
import time
import gym
from gym.spaces import Dict as GymDict, Box
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple
from my_env import Tools
from my_env.ToolConn import ToolConn
# from marllib.envs.base_env import ENV_REGISTRY
from my_env.channels import FreeFriis, OkumuraHata
from my_env.entities import BaseStation, UAVStation, UserEquipment
# from my_env.handlers.central import MComCentralHandler
from my_env.handlers.multi_agent import MComMAHandler
# from my_env.link import myLink
from my_env.movement import RandomUAVMove, RandomUEMove
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from marllib import marl

policy_mapping_dict = {
    "all_scenario": {
        "description": "myCommEnvMA all scenarios",
        "team_prefix": ("uavt1_", "uavt2_", "uavt3_", "uavt4_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class myCommEnvMA(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self.width, self.height, self.h_3d = 3000, 3000, 300   # 图大小
        self.NUM_UAV = 14
        self.NUM_UE = 100
        station_pos = [(0, 3000), (3000, 0), (0, 0)]
        self.uav_team_prefix = policy_mapping_dict["all_scenario"]["team_prefix"]
        uav_team_num = [2, 3, 4, 5]
        self.uav_team_id = [uav_team_num[0], uav_team_num[0] + uav_team_num[1], 
                            uav_team_num[0] + uav_team_num[1] + uav_team_num[2], 
                            uav_team_num[0] + uav_team_num[1] + uav_team_num[2] + uav_team_num[3]]
        config_BS = {"bw": 9e6, "freq": 2500, "tx": 30, "height": 80}
        config_UE = {"snr_th": 0.0085, "noise": 1e-9, "height": 1.5,}
        config_UAV = {"snr_bs_th": 120, "snr_uav_th": 0.22, "noise": 1e-9, "bw": 9e6, 
                      "freq": 2500, "tx": 0.3, "NUM_UAV": self.NUM_UAV, "NUM_BS": len(station_pos), 
                      "uav_team_id": self.uav_team_id, "uav_team_prefix": self.uav_team_prefix}
        
        self.my_seed = None
        reset_rng_episode = False       # 不重置随机数生成器，初始位置随机
        self.EP_MAX_TIME = 400         # 结束时间
        move_hori = list(itertools.product([-30, 0, 30], [-30, 0, 30], [0]))
        move_vert = list(itertools.product([0], [0], [-10, 10]))
        self.uav_action_lst = move_hori + move_vert
        # self.uav_action_lst = list(itertools.product([-30, 0, 30], [-30, 0, 30], [-10, 0, 10]))
        ue_width, ue_height = 500, 500  # 人范围比整图小多少
        
        config_RandomUEMove = {
            "move_ue_random": [-10, 0, 10],
            "UEPosLimit": [ue_width, self.width - ue_width, ue_height, self.height - ue_height],
            "reset_rng_episode": reset_rng_episode,
            "NUM_UE": self.NUM_UE,
        }
        config_Tools = {
            "move_ue_random": [-10, 0, 10],
            "UEPosLimit": [ue_width, self.width - ue_width, ue_height, self.height - ue_height],
            "reset_rng_episode": reset_rng_episode,
        }
        config_RandomUAVMove = {
            "UAVPosLimit": [self.width, self.height, self.h_3d],
            "reset_rng_episode": reset_rng_episode,
        }
        self.movement_ue = RandomUEMove(**config_RandomUEMove)      # UE移动方式
        self.movement_uav = RandomUAVMove(**config_RandomUAVMove)   # UAV移动方式
        # 规划UAV基站的资源给每个用户, 暂不考虑基站和UAV之间的  "scheduler": ResourceFair
        # "utility": BoundedLogUtility, # UE的QoE的量化方式？？

        stations = [BaseStation(bs_id, pos, **config_BS) for bs_id, pos in enumerate(station_pos)]
        users = [UserEquipment(ue_id, **config_UE) for ue_id in range(self.NUM_UE)]
        uavs = [UAVStation(uav_id, **config_UAV) for uav_id in range(self.NUM_UAV)]

        self.channel_ue = OkumuraHata() # OkumuraHata模型
        self.channel_bs = FreeFriis()   # FreeFriis模型 UAV-BS和UAV-UAV都用的这个
        # self.myLink = myLink()

        # define parameters that track the simulation's progress
        self.time = None
        self.closed = False

        # defines the simulation's overall basestations and UEs
        self.BS = {bs.bs_id: bs for bs in stations}
        self.UE = {ue.ue_id: ue for ue in users}
        self.NUM_BS = len(self.BS)

        # self.ax = plt.figure().add_subplot(projection='3d') # render
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax1 = self.fig.add_subplot(122)
        
        ### CHANGE ###
        self.UAV = {uav.uav_type + str(uav.uav_id): uav for uav in uavs}
        
        self.handler = MComMAHandler
        self.ori_action_space = self.handler.action_space(self)
        self.ori_observation_space = self.handler.observation_space(self)

        self.agents = list(self.UAV.keys())
        self.num_agents = len(self.agents)
        self.action_space = self.ori_action_space[self.agents[0]]
        self.observation_space = GymDict({"obs": self.ori_observation_space[self.agents[0]],
                                          "state": self.handler.state_space(self)})
        # self.action_space = self.ori_action_space
        # self.observation_space = self.ori_observation_space
        self.env_config = env_config
        # print()
        self.myTools = Tools.MyPosition(**config_Tools)
        self.myToolConn = ToolConn(snr_bs_th=120, snr_uav_th=0.22)

        self.curriculum = 0


    def reset(self, *, seed=None, options=None):
        """Reset env to starting state. Return the initial obs and info."""
        # super().reset(seed=seed)
        self.log_UEMaxDR = 0
        self.NUM_UE_CONN = 0
        self.UE_DR_TOTAL = 0
        self.reward_render = 0
        if seed is not None:
            self.my_seed = seed

        # reset time
        self.time = 0
        self.closed = False

        # reset state kept by arrival pattern, channel, scheduler, etc.
        # self.channel_ue.reset()
        # self.channel_bs.reset()
        if self.my_seed:
            self.movement_ue.reset(self.my_seed + 1)    # 重置movement_ue的rng
            self.movement_uav.reset(self.my_seed + 2)
        else:
            self.movement_ue.reset()    # 重置movement_ue的rng
            self.movement_uav.reset()

        # generate new initial positons of UEs and UAVs
        for ue in self.UE.values():
            self.movement_ue.initial_position(ue)
        for uav in self.UAV.values():
            self.movement_uav.initial_position(uav)
        # self.movement_uav.initial_position_check(self.UAV, self.BS)

        info = {}

        self.myTools.reset(self, self.my_seed)
        self.myToolConn.reset(self.myTools, self)

        ### CHANGE ###
        the_global_state = self.handler.my_global_state(self.myToolConn.ObsDict)
        original_obs = self.handler.observation(self.myToolConn.ObsDict, self.agents)
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[name], 
                         "state": the_global_state,}

        # # LLM ChatGPT
        # UE_pos = (self.myTools.UE_position[:, :2] - 500) // 250
        # UE_pos = np.clip(UE_pos, 0, 7).astype(int)
        # UE_grid = np.unique(UE_pos, axis=0, return_counts=True)
        # Str_UE_grid = np.zeros((8, 8))
        # for i in range(UE_grid[0].shape[0]):
        #     a = UE_grid[0][i, :]
        #     Str_UE_grid[a[0], a[1]] = UE_grid[1][i]
        # Str_UE_grid = str(Str_UE_grid.astype(int))
        # print(Str_UE_grid)
            
        # # print(self.myTools.UE_position)
        # aa = str(self.myTools.UE_position[:, :2].astype(int))
        # print(aa)
        return obs

    def step(self, actions: List[int]):
        assert not self.closed, "step() called on terminated episode"
        # if self.time ==3:
        #     print(int(self.curriculum/100), end=' ')

        # apply handler to transform actions to expected shape
        self.myTools.step(self, actions)
        self.myToolConn.reset(self.myTools, self)
        rewards, self.NUM_UE_CONN, self.UE_DR_TOTAL, self.all_reward_render = self.handler.reward(self.myToolConn, 
                                                                          self.myTools.O_DistUAV_UAV,
                                                                          self.agents, self.uav_team_id, 
                                                                          self.curriculum)

        self.log_UEMaxDR += np.sum(self.myToolConn.ObsDict['UE_MaxDR'])
        # self.reward_render += sum(rewards.values())
        self.reward_render += self.all_reward_render[0]
        self.time += 1

        ### CHANGE ###
        the_global_state = self.handler.my_global_state(self.myToolConn.ObsDict)
        original_obs = self.handler.observation(self.myToolConn.ObsDict, self.agents)

        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[name], 
                         "state": the_global_state,}

        info = {}

        # check whether episode is done & close the environment
        if self.time >= self.EP_MAX_TIME:
            self.closed = True
        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        # terminated = self.closed
        terminated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}
        terminated["__all__"] = self.closed
        # truncated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}

        return obs, rewards, terminated, info

    def render(self):
        if self.time > 390:
            sys.exit()
        # self.ax.cla()
            
        # # move user equipments around; update positions of UEs TODO
        # for i in range(len(self.UAV)):
        #     uav_i = self.UAV[self.agents[i]]
        #     uav_i.x, uav_i.y, uav_i.height = self.myTools.UAV_position[i, :]
        #     # self.movement_uav.move(self.UAV[self.agents[i]], actions[self.agents[i]])   ### CHANGE ###
        # for i in range(len(self.UE)):       # 按理说应该算完reward后人动
        #     self.UE[i].x, self.UE[i].y = self.myTools.UE_position[i, :2]
        #     # self.movement_ue.move(self.UE[i], self.myTools.UE_Actions[i, :])

        self.fig.clf()
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax1 = self.fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1)

        self.ax = self.get_render_axis(self.ax)
        self.ax.view_init(elev=90, azim=0)
        self.ax1 = self.get_render_axis(self.ax1)
        self.ax1.view_init(elev=0, azim=0)
        plt.pause(0.001)
        self.fig.suptitle("t=" + str(self.time) + "|R=" + str(round(self.log_UEMaxDR, 3)) + "|AvgR=" + str(round(self.log_UEMaxDR/self.time, 2)) + \
                  "|Num=" + str(self.NUM_UE_CONN) + "|StepR=" + str(round(self.UE_DR_TOTAL, 2)) + "|AllNum=" + str(round(self.reward_render / self.time, 2)) + 
                  "\n" + str(round(self.all_reward_render[0], 1)) + " | " + str(round(self.all_reward_render[1], 3)) + 
                  " | " + str(round(self.all_reward_render[2], 3)) + " | " + str(round(self.all_reward_render[3], 3)) + 
                  " | " + str(round(self.all_reward_render[4], 3)) + " | " + str(round(self.all_reward_render[5], 3)) + 
                  " | " + str(round(self.all_reward_render[6], 3)))
        plt.savefig('results/imgs/ma' + str(self.time))
        print("Saved to results/imgs/ma" + str(self.time) + " AllNum = " + str(round(self.reward_render / self.time, 2)))
        return
    
    def get_render_axis(self, ax):
        normFactor = max(self.width, self.height, self.h_3d)
        ax.scatter( self.myToolConn.ObsDict['Locs_UE'][:, 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UE'][:, 1] * normFactor, 
                    self.UE[0].height, marker='o', s=1)
        ax.scatter( self.myToolConn.ObsDict['Locs_UAV'][:self.uav_team_id[0], 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][:self.uav_team_id[0], 1] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][:self.uav_team_id[0], 2] * normFactor,
                    marker='x', color='greenyellow')
        ax.scatter( self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[0]:self.uav_team_id[1], 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[0]:self.uav_team_id[1], 1] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[0]:self.uav_team_id[1], 2] * normFactor,
                    marker='x', color='orange')
        ax.scatter( self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[1]:self.uav_team_id[2], 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[1]:self.uav_team_id[2], 1] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[1]:self.uav_team_id[2], 2] * normFactor,
                    marker='x', color='aqua')
        ax.scatter( self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[2]:, 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[2]:, 1] * normFactor, 
                    self.myToolConn.ObsDict['Locs_UAV'][self.uav_team_id[2]:, 2] * normFactor,
                    marker='x', color='deeppink')
        ax.scatter( self.myToolConn.ObsDict['Locs_BS'][:, 0] * normFactor, 
                    self.myToolConn.ObsDict['Locs_BS'][:, 1] * normFactor, 
                    self.BS[0].height, marker='s')
        
        UEid = np.where(np.array(self.myToolConn.UE_UAVid) > -1)[0]
        for i in UEid:
            uav_i = self.myToolConn.UE_UAVid[i]
            ue_pos = self.myTools.UE_position[i, :]
            uav_pos = self.myTools.UAV_position[uav_i, :]
            ax.plot3D(  [ue_pos[0], uav_pos[0]], 
                        [ue_pos[1], uav_pos[1]], 
                        [ue_pos[2], uav_pos[2]], 
                        c='r')
        
        for i in range(len(self.myToolConn.Link_backward)):
            con_uavs = self.myToolConn.Link_backward[i]
            if isinstance(con_uavs, list) and len(con_uavs) > 0:
                uav_pos = self.myTools.UAV_position[i]
                j = con_uavs[-1]
                # for j in con_uavs:
                j_pos = self.myTools.UAV_position[j]
                ax.plot3D(  [uav_pos[0], j_pos[0]], 
                            [uav_pos[1], j_pos[1]], 
                            [uav_pos[2], j_pos[2]], 
                            c='blue')
                
        for i in range(len(self.myToolConn.Link_BS)):
            if self.myToolConn.Link_BS[i] > -1:
                uav_pos = self.myTools.UAV_position[i]
                bs_pos = self.myToolConn.Link_BS[i]
                bs_pos = self.myTools.BS_position[bs_pos, :]
                ax.plot3D(  [uav_pos[0], bs_pos[0]], 
                            [uav_pos[1], bs_pos[1]], 
                            [uav_pos[2], bs_pos[2]], 
                            c='green')

        ax.set_zlim(0, self.h_3d)    # 有效
        return ax

    
    ### CHANGE ###
    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.EP_MAX_TIME,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
    
    def close(self):
        pass
        
if __name__ == '__main__':
    a = myCommEnvMA(0)

    target_pos = [[500, 500], [750, 750], [1250, 750], [1750, 750], [2250, 750],
                  [750, 1250], [1250, 1250], [1750, 1250], [2250, 1250],
                  [750, 1750], [1250, 1750], [1750, 1750], [2250, 1750], 
                  [750, 2250]
                  ]
    target_pos = np.array(target_pos)
    # height = 

    for i in range(1):
        a.reset()
        for j in range(395):
            action_s = {}
            for k in range(a.myToolConn.myTools.UAV_position.shape[0]):
                up = a.myToolConn.myTools.UAV_position[k, :2]
                tp = target_pos[k, :2]
                tp_ac = np.clip(tp - up, -30, 30).astype(int)
                tp_ac = np.where(tp_ac ** 2 == 900, tp_ac, 0)
                for l in range(len(a.uav_action_lst)):
                    if tp_ac[0] == a.uav_action_lst[l][0] and tp_ac[1] == a.uav_action_lst[l][1]:
                        action_s[a.agents[k]] = l
                        break

                # print()
            
            dummy_action = a.ori_action_space.sample()
            dummy_action = action_s
        #     # # dummy_action = np.array([10, 22, 20, 23, 20,  3,  6, 19, 16, 16, 12,  1,  9])
            obs, reward, terminated, info = a.step(dummy_action)
            a.render()
    #     # # a.step()

#     # register new env
#     ENV_REGISTRY["myCommEnvMA"] = myCommEnvMA
#     # initialize env
#     env = marl.make_env(environment_name="myCommEnvMA", map_name="myCommEnvMA", abs_path="myCommEnvMA.yaml")
#     # pick mappo algorithms
#     mappo = marl.algos.mappo(hyperparam_source="test")
#     # customize model
#     model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
#     # start learning
#     mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
#               num_workers=2, share_policy='all', checkpoint_freq=50)
    print()