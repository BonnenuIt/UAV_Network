from typing import Tuple
from shapely.geometry import Point

class BaseStation:
    def __init__(
        self,
        bs_id: int,
        pos: Tuple[float, float],
        bw: float,
        freq: float,
        tx: float,
        height: float,
    ):
        # BS ID should be final, i.e., BS ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.bs_id = bs_id
        self.x, self.y = pos
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        self.height = height  # in m

class UserEquipment:
    def __init__(
        self,
        ue_id: int,
        # velocity: float,
        snr_th: float,
        noise: float,
        height: float,
    ):
        # UE ID should be final, i.e., UE ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.ue_id = ue_id
        # self.velocity: float = velocity # 速度
        self.snr_th = snr_th     # SNR阈值，多少可通信UAV -> UE
        self.noise = noise
        self.height = height            # 高度

        self.x: float = None
        self.y: float = None
        # self.stime: int = None
        # self.extime: int = None
        self.link2UAV = None
        self.linkDataRate = 0

class UAVStation:
    def __init__(
        self,
        uav_id: int,
        # velocity: float,
        snr_bs_th: float,
        snr_uav_th: float,
        noise: float,
        bw: float,
        freq: float,
        tx: float,
        # height: float,
        NUM_UAV: int,
        NUM_BS: int,
        uav_team_id,
        uav_team_prefix,
    ):
        # "uav": {
        #         "velocity": 2,
        #         "snr_tr": 2e-8,
        #         "noise": 1e-9,
        #         "bw": 9e6,
        #         "freq": 2500,
        #         "tx": 30, 
        #     },
        # UAV ID should be final, i.e., UAV ID must be unique
        # NOTE: cannot use Final typing due to support for Python 3.7
        self.uav_id = uav_id
        # self.velocity: float = velocity # 速度
        # self.snr_threshold = snr_tr     # SNR阈值，多少可通信
        self.snr_bs_th = snr_bs_th
        self.snr_uav_th = snr_uav_th
        self.noise = noise
        self.bw = bw  # in Hz
        self.frequency = freq  # in MHz
        self.tx_power = tx  # in dBm
        # self.height = height            # 高度

        self.x: float = None
        self.y: float = None
        self.height: float = None
        # self.stime: int = None
        # self.extime: int = None
        self.link2BS = None
        self.linkQuality = 0

        if uav_id < uav_team_id[0]:
            self.uav_type = uav_team_prefix[0]
        elif uav_id < uav_team_id[1]:
            self.uav_type = uav_team_prefix[1]
        elif uav_id < uav_team_id[2]:
            self.uav_type = uav_team_prefix[2]
        else:
            self.uav_type = uav_team_prefix[3]
