from math import sqrt
from typing import Dict, Tuple
import numpy as np
from my_env.entities import UAVStation, UserEquipment

class RandomUEMove():
    def __init__(self, UEPosLimit, move_ue_random, reset_rng_episode, NUM_UE):
        self.rng = None
        self.reset_rng_episode = reset_rng_episode  # 是否每个episode都重置随机数生成器

        self.xMin, self.xMax, self.yMin, self.yMax = UEPosLimit
        self.move_ue_random = move_ue_random
        self.UEPosLimit = UEPosLimit
        self.NUM_UE = NUM_UE

    def reset(self, seed=None) -> None:
        if self.reset_rng_episode or self.rng is None:  # 如果是实例初始化时，或者要求每个episode都重置时
            self.rng = np.random.default_rng(seed)      # 重置随机数生成器
        self.init_positions = self.RandLocGen(self.NUM_UE, self.UEPosLimit)
        
    def move(self, ue: UserEquipment, action: Tuple[float, float] = None):
        """Move UE a step towards the random waypoint."""
        if action is None:
            x_change, y_change = self.rng.choice(self.move_ue_random, 2)
        else:
            x_change, y_change = action
        ue.x = np.clip(ue.x + x_change, self.xMin, self.xMax)
        ue.y = np.clip(ue.y + y_change, self.yMin, self.yMax)

    def initial_position(self, ue: UserEquipment) -> Tuple[float, float]:
        """Return initial position of UE at the beginning of the episode."""
        # ue.x = self.rng.uniform(self.xMin, self.xMax)
        # ue.y = self.rng.uniform(self.yMin, self.yMax)
        # return x, y
        ue.x = self.init_positions[ue.ue_id, 0]
        ue.y = self.init_positions[ue.ue_id, 1]

    def RandLocGen(self, UE_num_total, UEPosLimit=[500, 2500, 500, 2500]):
        '''
        xMin, xMax, yMin, yMax
        '''
        if self.rng.integers(0, 2) == 0:    # 一半概率聚落正态，一半概率均匀分布
            return self.rng.uniform(low=UEPosLimit[0], high=UEPosLimit[1], size=(UE_num_total, 2)) # 没管x、y
        groups = self.rng.integers(3, 7)
        UE_num = int(UE_num_total / groups)
        for i in range(groups):
            banjing = self.rng.integers(400, 1500)
            center = np.array([self.rng.integers(1000, 2000), 
                            self.rng.integers(1000, 2000)])
            if i == 0:
                x = self.rng.normal(center, banjing / 2, size = (UE_num, 2))
            else:
                x = np.concatenate((x, self.rng.normal(center, banjing / 2, size = (UE_num, 2))))
        x_logit_x = np.logical_or(x[:, 0] < UEPosLimit[0], x[:, 0] > UEPosLimit[1])
        x_logit_y = np.logical_or(x[:, 1] < UEPosLimit[2], x[:, 1] > UEPosLimit[3])
        x_logit = np.logical_or(x_logit_x, x_logit_y).nonzero()[0]
        
        x = np.delete(x, x_logit, axis=0)
        del_num = UE_num_total - x.shape[0]
        uniform_x = self.rng.uniform(low=UEPosLimit[0], high=UEPosLimit[1], size=(del_num, 2)) # 没管x、y

        return np.concatenate((x, uniform_x))


class RandomUAVMove():
    def __init__(self, UAVPosLimit, reset_rng_episode):

        self.width, self.height, self.h_3d = UAVPosLimit # 图的总大小
        self.reset_rng_episode = reset_rng_episode  # 重置随机数生成器
        self.rng = None

    def reset(self, seed=None) -> None:
        if self.reset_rng_episode or self.rng is None:  # 如果是实例初始化时，或者要求每个episode都重置时
            self.rng = np.random.default_rng(seed)      # 重置随机数生成器

    def move(self, uav: UAVStation, action: Tuple[float, float, float]):
        """Move UAV."""
        assert len(action) == 3, "UAV move(action): len(action) should be 3"
        
        uav.x = np.clip(uav.x + action[0], 0, self.width)
        uav.y = np.clip(uav.y + action[1], 0, self.height)
        uav.height = np.clip(uav.height + action[2], 5, self.h_3d)    # 高度不为0，通信log10要求
    
    def initial_position(self, uav: UAVStation) -> Tuple[float, float, float]:
        """Return initial position of UAV at the beginning of the episode."""
        uav.x = self.rng.uniform(0, self.width)
        uav.y = self.rng.uniform(0, self.height)
        uav.height = self.rng.uniform(5, self.h_3d)

    def initial_position_check(self, uavs, BS):
        UAV_locs = np.array([[i.x, i.y, i.height] for i in uavs.values()])
        # BS_locs = np.array([[i.x, i.y, i.height] for i in BS.values()])
        UAV_BS_dist = np.zeros((len(uavs), len(BS)))
        for i in range(len(uavs)):      # 遍历每个UAV
            for j in range(len(BS)):   # 遍历每个BS
                uav_loc = UAV_locs[i, :]
                BS_loc = [BS[j].x, BS[j].y, BS[j].height]
                UAV_BS_dist[i, j] = self.distance_3d(uav_loc, BS_loc)
        
        t1_ids = np.argmin(UAV_BS_dist, axis=0) # 输出3个UAV序号
        t1_locs = UAV_locs[t1_ids, :]

        UAV_locs_rest = np.delete(UAV_locs, t1_ids, axis=0)
        UAV_BS_dist = np.delete(UAV_BS_dist, t1_ids, axis=0)
        UAV_BS_dist = np.min(UAV_BS_dist, axis=1)
        UAV_BS_dist_idx = np.argsort(UAV_BS_dist)

        # t2_ids = UAV_BS_dist_idx[:int(UAV_BS_dist_idx.shape[0]/2)]
        # t2_locs = UAV_locs[, :]
        t2_3_locs = UAV_locs_rest[UAV_BS_dist_idx, :]

        for i in uavs.values():
            i.x, i.y, i.height = None, None, None
            if i.uav_type == 'uavt1_':
                i.x, i.y, i.height = t1_locs[0, :]
                t1_locs = np.delete(t1_locs, 0, axis=0)
            elif i.uav_type == 'uavt2_':
                i.x, i.y, i.height = t2_3_locs[0, :]
                t2_3_locs = np.delete(t2_3_locs, 0, axis=0)
        
        for i in uavs.values():
            if i.uav_type == 'uavt3_':
                i.x, i.y, i.height = t2_3_locs[0, :]
                t2_3_locs = np.delete(t2_3_locs, 0, axis=0)
        
        return

    def distance_3d(self, p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


