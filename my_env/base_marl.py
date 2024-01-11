import copy
import itertools
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple
from my_env.channels import FreeFriis, OkumuraHata
from my_env.entities import BaseStation, UAVStation, UserEquipment
from my_env.handlers.central import MComCentralHandler
from my_env.handlers.multi_agent import MComMAHandler
from my_env.link import myLink
from my_env.movement import RandomUAVMove, RandomUEMove

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from pettingzoo import ParallelEnv

class myCommEnvMA(gymnasium.Env):
# class myCommEnvMA(ParallelEnv):
    def __init__(self):
        super().__init__()
        self.width, self.height, self.h_3d = 3000, 3000, 300   # 图大小
        self.NUM_UAV = 10
        self.NUM_UE = 50
        station_pos = [(0, 3000), (3000, 0), (0, 0)]
        config_BS = {"bw": 9e6, "freq": 2500, "tx": 30, "height": 80}
        config_UE = {"snr_th": 0.0024, "noise": 1e-9, "height": 1.5,}
        config_UAV = {"snr_bs_th": 150, "snr_uav_th": 0.4, "noise": 1e-9, "bw": 9e6, "freq": 2500, "tx": 0.3,}

        self.my_seed = 888
        reset_rng_episode = False       # 不重置随机数生成器，初始位置随机
        self.EP_MAX_TIME = 1000         # 结束时间
        move_hori = list(itertools.product([-30, 0, 30], [-30, 0, 30], [0]))
        move_vert = list(itertools.product([0], [0], [-10, 10]))
        self.uav_action_lst = move_hori + move_vert
        # self.uav_action_lst = list(itertools.product([-30, 0, 30], [-30, 0, 30], [-10, 0, 10]))
        ue_width, ue_height = 500, 500  # 人范围比整图小多少
        
        config_RandomUEMove = {
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
        self.myLink = myLink()

        # define parameters that track the simulation's progress
        self.time = None
        self.closed = False

        # defines the simulation's overall basestations and UEs
        self.BS = {bs.bs_id: bs for bs in stations}
        self.UE = {ue.ue_id: ue for ue in users}
        self.UAV = {uav.uav_id: uav for uav in uavs}
        # self.possible_agents = list(self.UAV.keys())
        # self.agents = self.UAV
        self.NUM_BS = len(self.BS)

        self.handler = MComMAHandler
        self.action_space = self.handler.action_space(self)
        self.observation_space = self.handler.observation_space(self)

        self.ax = plt.figure().add_subplot(projection='3d') # render

    def reset(self, *, seed=None, options=None):
        """Reset env to starting state. Return the initial obs and info."""
        super().reset(seed=seed)
        if seed is not None:
            self.my_seed = seed

        # reset time
        self.time = 0
        self.closed = False

        # reset state kept by arrival pattern, channel, scheduler, etc.
        # self.channel_ue.reset()
        # self.channel_bs.reset()
        self.movement_ue.reset(self.my_seed + 1)    # 重置movement_ue的rng
        self.movement_uav.reset(self.my_seed + 2)

        # generate new initial positons of UEs and UAVs
        for ue in self.UE.values():
            self.movement_ue.initial_position(ue)
        for uav in self.UAV.values():
            self.movement_uav.initial_position(uav)

        # info = {agent: {} for agent in self.UAV}
        info = {}

        self.myLink.reset(env=self)
        self.myLink.MyConnection(env=self)    # DataRateUE_UAV 取最大（utility）和索引（画link）
        UAV_Available_Lst = self.myLink.UAV_Available(self, self.UAV[0].snr_bs_th, self.UAV[0].snr_uav_th)
        # 判断用户连的uav可用不
        self.myLink.UE_ConnDataRate(self, UAV_Available_Lst)
        self.myLink.Obs_normalise(self)

        return self.handler.observation(self), info

    def step(self, actions: List[int]):
        assert not self.closed, "step() called on terminated episode"

        # apply handler to transform actions to expected shape
        actions = self.handler.action(self, actions)    # 每个UAV : 对应动作

        # move user equipments around; update positions of UEs TODO
        for i in range(len(self.UAV)):
            self.movement_uav.move(self.UAV[i], actions[i])
        for i in range(len(self.UE)):       # 按理说应该算完reward后人动
            self.movement_ue.move(self.UE[i], None)

        self.myLink.reset(env=self)
        self.myLink.MyConnection(env=self)    # DataRateUE_UAV 取最大（utility）和索引（画link）
        UAV_Available_Lst = self.myLink.UAV_Available(self, self.UAV[0].snr_bs_th, self.UAV[0].snr_uav_th)
        # 判断用户连的uav可用不
        self.myLink.UE_ConnDataRate(self, UAV_Available_Lst)
        self.myLink.Obs_normalise(self)

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)

        # update internal time of environment
        self.time += 1

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)

        info = self.myLink.ObsDict
        info['normFactor'] = self.myLink.normFactor

        # check whether episode is done & close the environment
        if self.time >= self.EP_MAX_TIME:
            self.closed = True
        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        # terminated = self.closed
        terminated = {uav.uav_id: self.closed for uav in self.UAV.values()}
        terminated["__all__"] = self.closed
        truncated = {uav.uav_id: self.closed for uav in self.UAV.values()}

        return observation, rewards, terminated, truncated, info

    def render(self):
        normFactor = max(self.width, self.height, self.h_3d)
        self.ax.cla()
        self.ax.scatter(self.myLink.ObsDict['Locs_UE'][:, 0] * normFactor, 
                        self.myLink.ObsDict['Locs_UE'][:, 1] * normFactor, 
                        self.UE[0].height, marker='o', s=1)
        self.ax.scatter(self.myLink.ObsDict['Locs_UAV'][:, 0] * normFactor, 
                        self.myLink.ObsDict['Locs_UAV'][:, 1] * normFactor, 
                        self.myLink.ObsDict['Locs_UAV'][:, 2] * normFactor,
                        marker='x')
        self.ax.scatter(self.myLink.ObsDict['Locs_BS'][:, 0] * normFactor, 
                        self.myLink.ObsDict['Locs_BS'][:, 1] * normFactor, 
                        self.BS[0].height, marker='s')
        for i in range(len(self.UE)):
            if self.UE[i].link2UAV != None:
                UAV_ue = self.UAV[self.UE[i].link2UAV[0]]
                self.ax.plot3D([self.UE[i].x, UAV_ue.x], 
                               [self.UE[i].y, UAV_ue.y], 
                               [self.UE[i].height, UAV_ue.height], 
                               c='r')
        
        for i in range(len(self.UAV)):
            if self.UAV[i].link2BS != None:
                if len(self.UAV[i].link2BS) == 1:   # 直连基站
                    UAV_bs = self.BS[self.UAV[i].link2BS[0]]
                    self.ax.plot3D([self.UAV[i].x, UAV_bs.x], 
                                    [self.UAV[i].y, UAV_bs.y], 
                                    [self.UAV[i].height, UAV_bs.height], 
                                    c='green')
                else:
                    UAV_uav = self.UAV[self.UAV[i].link2BS[-1]]
                    self.ax.plot3D([self.UAV[i].x, UAV_uav.x], 
                                    [self.UAV[i].y, UAV_uav.y], 
                                    [self.UAV[i].height, UAV_uav.height], 
                                    c='blue')
                    
        self.ax.set_zlim(0, self.h_3d)    # 有效
        self.ax.view_init(elev=75, azim=0)
        plt.pause(0.03)
        return
        