from math import sqrt
import numpy as np
from my_env.entities import BaseStation, UserEquipment

class BaseChannel():
    def __init__(self) -> None:
        self.EPSILON = 1e-7

    def distance_3d_km(self, p1, p2):
        return (sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) + self.EPSILON) / 1000
    
    def distance_2d_km(self, p1, p2):
        return (sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) + self.EPSILON) / 1000

    def datarate(self, bs: BaseStation, ue: UserEquipment, snr: float, snr_th: float):   # bps
        """Calculate max. data rate for transmission between BS and UE."""
        if snr > snr_th:
            return bs.bw * np.log2(1 + snr)
        return 0.0
    
    def snr(self, bs: BaseStation, ue: UserEquipment):
        """Calculate SNR for transmission between BS and UE."""
        loss = self.power_loss(bs, ue)
        power = 10 ** ((bs.tx_power - loss) / 10)
        snr = power / ue.noise
        return snr

class OkumuraHata(BaseChannel):
    def __init__(self) -> None:
        super(OkumuraHata, self).__init__()

    def power_loss(self, bs: BaseStation, ue: UserEquipment):
        # distance = (bs.point.distance(ue.point) + EPSILON)/1000
        distance = self.distance_2d_km([bs.x, bs.y, bs.height], [ue.x, ue.y, ue.height])
        self.dist = distance

        ch = (
            0.8
            + (1.1 * np.log10(bs.frequency) - 0.7) * ue.height
            - 1.56 * np.log10(bs.frequency)
        )
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(bs.frequency) - 13.82 * np.log10(bs.height)   # 市区传播公式损耗公式
        )
        tmp_2 = 44.9 - 6.55 * np.log10(bs.height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance)

class FreeFriis(BaseChannel):
    def __init__(self, d_thres=0.5) -> None:
        super(FreeFriis, self).__init__()
        self.d_thres = d_thres

    def power_loss(self, bs: BaseStation, ue: UserEquipment):
        # distance = (bs.point.distance(ue.point) + EPSILON) / 1000
        distance = self.distance_3d_km([bs.x, bs.y, bs.height], [ue.x, ue.y, ue.height])
        self.dist = self.distance_2d_km([bs.x, bs.y, bs.height], [ue.x, ue.y, ue.height])
        # if distance < self.d_thres:
        return 32.5 + 20 * np.log10(2400) + 20 * np.log10(distance)    # 2.4GHz常用通信频段 140m波长
        # else:
        #     return 1e10
