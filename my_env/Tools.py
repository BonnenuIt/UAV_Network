
import numpy as np
from scipy.spatial.distance import pdist, squareform

class MyPosition():
    def __init__(self, UEPosLimit, move_ue_random, reset_rng_episode):
        self.reset_rng_episode = reset_rng_episode  # 重置随机数生成器
        self.rng = None

        self.xMin, self.xMax, self.yMin, self.yMax = UEPosLimit
        self.move_ue_random = move_ue_random

    def reset(self, env, seed=None):
        if self.reset_rng_episode or self.rng is None:  # 如果是实例初始化时，或者要求每个episode都重置时
            self.rng = np.random.default_rng(seed)      # 重置随机数生成器
        
        self.UAV_position = []
        self.UE_position = []
        self.BS_position = []

        self.UAV_UAV_2d = None
        self.UE_UAV_2d = None

        self.BS_UAV_3d = None
        self.UAV_UAV_3d = None

        for ue in env.UE.values():
            self.UE_position.append([ue.x, ue.y, ue.height])

        for uav in env.UAV.values():
            self.UAV_position.append([uav.x, uav.y, uav.height])

        for bs in env.BS.values():
            self.BS_position.append([bs.x, bs.y, bs.height])

        self.UE_position = np.array(self.UE_position)
        self.UAV_position = np.array(self.UAV_position)
        self.BS_position = np.array(self.BS_position)

        self.calDist(env)
        self.calComm(env)
        
        return
    
    def step(self, env, actions):
        self.UAV_move(actions, env=env)
        self.UE_move()

        self.calDist(env)
        self.calComm(env)
        
        return
    
    def UAV_move(self, actions, env):
        assert len(actions) == env.NUM_UAV, "Number of actions must equal overall UAVs."
        myActions = []
        for i in env.UAV:
            myActions.append(env.uav_action_lst[actions[i]])
        myActions = np.array(myActions)

        self.UAV_position = self.UAV_position + myActions
        self.UAV_position[:, 0] = np.clip(self.UAV_position[:, 0], 0, env.width)
        self.UAV_position[:, 1] = np.clip(self.UAV_position[:, 1], 0, env.height)
        self.UAV_position[:, 2] = np.clip(self.UAV_position[:, 2], 5, env.h_3d)
        return
    
    def UE_move(self):
        UE_Actions = self.rng.choice(self.move_ue_random, size=(self.UE_position.shape[0], 2))
        self.UE_position[:, :2] = self.UE_position[:, :2] + UE_Actions
        
        self.UE_position[:, 0] = np.clip(self.UE_position[:, 0], self.xMin, self.xMax)
        self.UE_position[:, 1] = np.clip(self.UE_position[:, 1], self.yMin, self.yMax)

        self.UE_Actions = UE_Actions


    def calDist(self, env):
        UAV_UE_position_2d = np.concatenate((self.UAV_position, self.UE_position))[:, :2]
        UAV_UE_Dists_2d = squareform(pdist(UAV_UE_position_2d)) / 1000
        self.UAV_UAV_2d = UAV_UE_Dists_2d[:env.NUM_UAV, :env.NUM_UAV]
        self.UE_UAV_2d = UAV_UE_Dists_2d[env.NUM_UAV:, :env.NUM_UAV]

        UAV_BS_position_3d = np.concatenate((self.UAV_position, self.BS_position))
        UAV_BS_Dists_3d = squareform(pdist(UAV_BS_position_3d)) / 1000
        self.UAV_UAV_3d = UAV_BS_Dists_3d[:env.NUM_UAV, :env.NUM_UAV]
        self.BS_UAV_3d = UAV_BS_Dists_3d[env.NUM_UAV:, :env.NUM_UAV]

        return
    
    def calComm(self, env):
        self.O_DistUAV_UAV = self.UAV_UAV_2d.copy()
        self.O_DistUE_UAV = self.UE_UAV_2d.copy()

        self.BS_UAV_Loss = self.Friis_power_loss(self.BS_UAV_3d)
        self.UAV_UAV_Loss = self.Friis_power_loss(self.UAV_UAV_3d)

        UAV_Height = self.UAV_position[:, 2]
        UAV_Height = np.expand_dims(UAV_Height,0).repeat(env.NUM_UE, axis=0)
        self.UE_UAV_Loss = self.OHata_power_loss(self.UE_UAV_2d, env.UAV['uavt1_0'].frequency, 
                                                UAV_Height, 
                                                self.UE_position[0, 2])
        
        self.O_ConnBS_UAV = self.snr(self.BS_UAV_Loss, env.BS[0].tx_power, env.UAV['uavt1_0'].noise)
        self.O_ConnUAV_UAV = self.snr(self.UAV_UAV_Loss, env.UAV['uavt1_0'].tx_power, env.UAV['uavt1_0'].noise)
        self.O_ConnUE_UAV = self.snr(self.UE_UAV_Loss, env.UAV['uavt1_0'].tx_power, env.UE[0].noise)
        self.O_DataRateUE_UAV = self.datarate(env.UAV['uavt1_0'].bw, self.O_ConnUE_UAV, env.UE[0].snr_th)   # uav, ue, snr, ue.snr_th


    def datarate(self, bs_bw, snr, snr_th):   # bps
        """Calculate max. data rate for transmission between BS and UE."""
        snr = np.where(snr > snr_th, snr, 0)
        return bs_bw * np.log2(1 + snr)
    
    def Friis_power_loss(self, distance):
        # 2.4GHz常用通信频段 140m波长
        return 32.5 + 20 * np.log10(2400) + 20 * np.log10(distance + 1e-10)
    
    def OHata_power_loss(self, distance_2d, bs_frequency, bs_height, ue_height):
        # 市区传播公式损耗公式
        ch = (0.8 + (1.1 * np.log10(bs_frequency) - 0.7) * ue_height 
              - 1.56 * np.log10(bs_frequency))
        tmp_1 = (69.55 - ch + 26.16 * np.log10(bs_frequency) - 13.82 
                 * np.log10(bs_height))
        tmp_2 = 44.9 - 6.55 * np.log10(bs_height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance_2d + 1e-10)
    
    def snr(self, loss, bs_tx_power, ue_noise):
        """Calculate SNR for transmission between BS and UE."""
        power = 10 ** ((bs_tx_power - loss) / 10)
        snr = power / ue_noise
        return snr
    
