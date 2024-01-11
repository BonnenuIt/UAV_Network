import copy
import itertools
import gymnasium
from gymnasium.spaces import Dict as GymDict, Box
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple
# from marllib.envs.base_env import ENV_REGISTRY
from my_env.channels import FreeFriis, OkumuraHata
from my_env.entities import BaseStation, UAVStation, UserEquipment
# from my_env.handlers.central import MComCentralHandler
from my_env.handlers.multi_agent import MComMAHandler
from my_env.link import myLink
from my_env.movement import RandomUAVMove, RandomUEMove
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from marllib import marl

policy_mapping_dict = {
    "all_scenario": {
        "description": "myCommEnvMA all scenarios",
        "team_prefix": ("uavt1_", "uavt2_", "uavt3_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class myCommEnvMA(gymnasium.Env):
    def __init__(self, env_config):
        super().__init__()
        self.width, self.height, self.h_3d = 3000, 3000, 300   # 图大小
        self.NUM_UAV = 10
        self.NUM_UE = 50
        station_pos = [(0, 3000), (3000, 0), (0, 0)]
        config_BS = {"bw": 9e6, "freq": 2500, "tx": 30, "height": 80}
        config_UE = {"snr_th": 0.01, "noise": 1e-9, "height": 1.5,}
        config_UAV = {"snr_bs_th": 120, "snr_uav_th": 0.22, "noise": 1e-9, "bw": 9e6, 
                      "freq": 2500, "tx": 0.3, "NUM_UAV": self.NUM_UAV, "NUM_BS": len(station_pos)}

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
        self.NUM_BS = len(self.BS)

        self.ax = plt.figure().add_subplot(projection='3d') # render
        
        ### CHANGE ###
        self.UAV = {uav.uav_type + str(uav.uav_id): uav for uav in uavs}
        
        self.handler = MComMAHandler
        self.ori_action_space = self.handler.action_space(self)
        self.ori_observation_space = self.handler.observation_space(self)

        self.agents = list(self.UAV.keys())
        self.num_agents = len(self.agents)
        self.action_space = self.ori_action_space[self.agents[0]]
        self.action_space = gymnasium.spaces.Discrete(11)
        # self.observation_space = GymDict({"obs": self.ori_observation_space[self.agents[0]],
        #                                   "state": self.handler.state_space(self)})
        # self.action_space = self.ori_action_space
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(66,), dtype=np.float64)
        self.env_config = env_config
        # print()

    def reset(self, *, seed=None, options=None):
        """Reset env to starting state. Return the initial obs and info."""
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
        self.movement_uav.initial_position_check(self.UAV, self.BS)

        info = {}

        self.myLink.reset(env=self)
        self.myLink.MyConnection(env=self)    # DataRateUE_UAV 取最大（utility）和索引（画link）
        UAV_Available_Lst = self.myLink.UAV_Available(self, 
                                                      self.UAV['uavt1_0'].snr_bs_th, 
                                                      self.UAV['uavt1_0'].snr_uav_th) ### CHANGE ###
        # 判断用户连的uav可用不
        self.myLink.UE_ConnDataRate(self, UAV_Available_Lst)
        self.myLink.Obs_normalise(self)


        ### CHANGE ###
        the_global_state = self.handler.my_global_state(self)
        original_obs = self.handler.observation(self)
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[name], 
                         "state": the_global_state,}

        return obs, info

    def step(self, actions: List[int]):
        assert not self.closed, "step() called on terminated episode"

        # apply handler to transform actions to expected shape
        actions = self.handler.action(self, actions)    # 每个UAV : 对应动作

        # move user equipments around; update positions of UEs TODO
        for i in range(len(self.UAV)):
            self.movement_uav.move(self.UAV[self.agents[i]], actions[self.agents[i]])   ### CHANGE ###
        for i in range(len(self.UE)):       # 按理说应该算完reward后人动
            self.movement_ue.move(self.UE[i], None)

        self.myLink.reset(env=self)
        self.myLink.MyConnection(env=self)    # DataRateUE_UAV 取最大（utility）和索引（画link）
        UAV_Available_Lst = self.myLink.UAV_Available(self, 
                                                      self.UAV['uavt1_0'].snr_bs_th, 
                                                      self.UAV['uavt1_0'].snr_uav_th) ### CHANGE ###
        # 判断用户连的uav可用不
        self.myLink.UE_ConnDataRate(self, UAV_Available_Lst)
        self.myLink.Obs_normalise(self)

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)
        self.my_rewards = rewards

        # update internal time of environment
        self.time += 1

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        # observation = self.handler.observation(self)
        
        ### CHANGE ###
        the_global_state = self.handler.my_global_state(self)
        original_obs = self.handler.observation(self)
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[name], 
                         "state": the_global_state,}

        info = self.myLink.ObsDict
        info['normFactor'] = self.myLink.normFactor

        # check whether episode is done & close the environment
        if self.time >= self.EP_MAX_TIME:
            self.closed = True
        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        # terminated = self.closed
        terminated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}
        terminated["__all__"] = self.closed
        truncated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}

        return obs, rewards, terminated, truncated, info

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
                this_uav = self.UE[i].link2UAV[0]
                UAV_ue = self.UAV[self.agents[this_uav]]
                self.ax.plot3D([self.UE[i].x, UAV_ue.x], 
                               [self.UE[i].y, UAV_ue.y], 
                               [self.UE[i].height, UAV_ue.height], 
                               c='r')
        
        for i in range(len(self.UAV)):
            that_uav = self.UAV[self.agents[i]]
            if that_uav.link2BS != None:
                if len(that_uav.link2BS) == 1:   # 直连基站
                    UAV_bs = self.BS[that_uav.link2BS[0]]
                    self.ax.plot3D([that_uav.x, UAV_bs.x], 
                                    [that_uav.y, UAV_bs.y], 
                                    [that_uav.height, UAV_bs.height], 
                                    c='green')
                else:
                    UAV_uav = self.UAV[self.agents[that_uav.link2BS[-1]]]
                    self.ax.plot3D([that_uav.x, UAV_uav.x], 
                                    [that_uav.y, UAV_uav.y], 
                                    [that_uav.height, UAV_uav.height], 
                                    c='blue')
                    
        self.ax.set_zlim(0, self.h_3d)    # 有效
        self.ax.view_init(elev=90, azim=0)
        # plt.pause(0.001)
        rewards_show = round(sum(self.my_rewards.values()), 3)
        plt.title("reward =" + str(rewards_show))
        # plt.pause(0.03)
        return
    
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
        
# if __name__ == '__main__':
#     a = myCommEnvMA(0)
#     a.reset()

#     # dummy_action = a.ori_action_space.sample()
#     # # dummy_action = np.array([10, 22, 20, 23, 20,  3,  6, 19, 16, 16, 12,  1,  9])
#     # obs, reward, terminated, truncated, info = a.step(dummy_action)
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