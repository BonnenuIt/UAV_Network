import numpy as np

class myLink:
    def __init__(self):
        pass

    def reset(self, env):
        self.ConnBS_UAV = None
        self.ConnUAV_UAV = None
        self.DataRateUE_UAV = None
        self.UAV_Available_Lst_onehot = None
        self.UE_MaxDR = None
        self.UE_UAVid = None
        self.ObsDict = None

        for i in env.UAV.values():
            i.link2BS = None
            i.linkQuality = 0
        for i in env.UE.values():
            i.link2UAV = None
            i.linkDataRate = 0
        
    def MyConnection(self, myTools):
        self.DistUAV_UAV = myTools.O_DistUAV_UAV

        self.ConnBS_UAV = myTools.O_ConnBS_UAV
        self.ConnUAV_UAV = myTools.O_ConnUAV_UAV
        self.ConnUE_UAV = myTools.O_ConnUE_UAV
        self.DataRateUE_UAV = myTools.O_DataRateUE_UAV


    def MyConnection_old(self, env):
        """Connection can be established if SNR exceeds threshold of UE."""
        ConnUAV_BS = []
        ConnUAV_UAV = []
        DistUAV_UAV = []
        ConnUAV_UE = []
        DataRateUAV_UE = []

        for uav in env.UAV.values():
            ConnUAV_BS_i = []
            ConnUAV_UAV_i = []
            DistUAV_UAV_i = []
            ConnUAV_UE_i = []
            DataRateUAV_UE_i = []
            for bs in env.BS.values():
                snr = env.channel_bs.snr(bs, uav)   # BS -> UAV 注意顺序
                ConnUAV_BS_i.append(snr)
            for uav_j in env.UAV.values():
                snr = env.channel_bs.snr(uav, uav_j)
                DistUAV_UAV_i.append(env.channel_bs.dist)
                ConnUAV_UAV_i.append(snr)
            for ue in env.UE.values():
                snr = env.channel_ue.snr(uav, ue)   # UAV -> UE 注意顺序
                dr = env.channel_ue.datarate(uav, ue, snr, ue.snr_th)
                ConnUAV_UE_i.append(snr)
                DataRateUAV_UE_i.append(dr)
            
            ConnUAV_BS.append(ConnUAV_BS_i)
            ConnUAV_UAV.append(ConnUAV_UAV_i)
            DistUAV_UAV.append(DistUAV_UAV_i)
            ConnUAV_UE.append(ConnUAV_UE_i)
            DataRateUAV_UE.append(DataRateUAV_UE_i)
        ConnUAV_BS = np.array(ConnUAV_BS).T   # (BS, UAV)
        ConnUAV_UAV = np.array(ConnUAV_UAV).T   # (UAV, UAV)
        ConnUAV_UE = np.array(ConnUAV_UE).T   # (UE, UAV)
        DataRateUAV_UE = np.array(DataRateUAV_UE).T   # (UE, UAV)
        self.DistUAV_UAV = np.array(DistUAV_UAV).T

        self.ConnBS_UAV = ConnUAV_BS
        self.ConnUAV_UAV = ConnUAV_UAV
        self.ConnUE_UAV = ConnUAV_UE
        self.DataRateUE_UAV = DataRateUAV_UE
        return

    def UAV_Available(self, env, snr_bs_th, snr_uav_th):    # ConnBS_UAV - (BS, UAV)
        UAV_Available_Lst = []
        uav_all_id = list(env.UAV.keys())
        # 无人机直连上基站
        for i in range(self.ConnBS_UAV.shape[1]):        # 遍历每个UAV
            for j in range(self.ConnBS_UAV.shape[0]):    # 遍历每个BS
                if self.ConnBS_UAV[j, i] >= snr_bs_th:  # 不存在一个无人机能连上多个基站的情况。（cell蜂窝）
                    UAV_Available_Lst.append(i)
                    env.UAV[uav_all_id[i]].link2BS = [j]
                    env.UAV[uav_all_id[i]].linkQuality = np.inf
                    break

        # 无人机通过中继连上基站
        
        for i in range(len(uav_all_id)):
            if i == len(UAV_Available_Lst):
                break
            uav_id = UAV_Available_Lst[i]           # 遍历UAV_Available_Lst中连上的UAV 编号uav_id
            uav_conn = self.ConnUAV_UAV[uav_id]     # 这个连基站的UAV所连的UAV
            for j in range(uav_conn.shape[0]):      # 遍历所有UAV
                if j == uav_id or env.UAV[uav_all_id[j]].linkQuality == np.inf:   
                    # 连自己的和直连BS的j去除, linkQuality最大即跳数*1e6，np.inf即直连
                    continue
                # if uav_conn[j] >= snr_th and (j not in UAV_Available_Lst):
                if uav_conn[j] >= snr_uav_th:           # uav_id连上了编号j
                    j_link2BS = env.UAV[uav_all_id[uav_id]].link2BS + [uav_id] # link2BS
                    j_linkQuality = uav_conn[j] + (1/len(j_link2BS)) * 1e6    # SNR + 跳数*1e6
                    if j_linkQuality > env.UAV[uav_all_id[j]].linkQuality:  # uav_id连上了编号j且链路比原本的好（跳数少，离得近）
                        UAV_Available_Lst.append(j)
                        env.UAV[uav_all_id[j]].link2BS = j_link2BS
                        env.UAV[uav_all_id[j]].linkQuality = j_linkQuality

        UAV_Available_Lst = sorted(list(set(UAV_Available_Lst)))    # 排序去重
        self.UAV_Available_Lst_onehot = self.List2OneHot1D(UAV_Available_Lst, uav_all_id)
        return UAV_Available_Lst
    
    def UE_ConnDataRate(self, env, UAV_Available_Lst):
        if UAV_Available_Lst == []:
            UE_MaxDR = np.zeros(self.DataRateUE_UAV.shape[0])
            UE_UAVid = [-1] * self.DataRateUE_UAV.shape[0]

        else:
            UE_Conn = self.DataRateUE_UAV[:, UAV_Available_Lst]  # [ue, avai uav]
            UE_MaxDR = np.max(UE_Conn, axis=1)
            UE_UAVid_fake = np.argmax(UE_Conn, axis=1)  # 去除后的索引值，不是对应的所有UAV，而只是可用UAV的索引
            UE_UAVid = []   # ue - uav id
            for i in range(len(UE_UAVid_fake)):
                if UE_MaxDR[i] < 1:     # 为0时（防止近似0大于0）
                    UE_UAVid.append(-1)
                else:
                    UE_UAVid.append(UAV_Available_Lst[UE_UAVid_fake[i]])
        
        # UE_MaxDR = np.around(UE_MaxDR)/10000
        # UE_Conn = np.around(UE_Conn)/10000
        # print(UE_Conn)
        # print(UE_MaxDR)
        # aaa= env.UE.values()
        for i in range(len(env.UE.values())):
            if UE_MaxDR[i] > 1:
                env.UE[i].link2UAV = [UE_UAVid[i]]
                env.UE[i].linkDataRate = UE_MaxDR[i]
        
        self.UE_MaxDR = UE_MaxDR
        self.UE_UAVid = UE_UAVid

    def List2OneHot1D(self, UAV_Available_Lst, uav_all_id):
        res = np.zeros(len(uav_all_id))
        for i in UAV_Available_Lst:
            res[i] += 1
        # print(UAV_Available_Lst, res)
        return res

    def Obs_normalise(self, env):
        self.ObsDict = {}
        self.ObsDict["ConnBS_UAV"] = self.MinMaxNorm(self.ConnBS_UAV, 0, 4000)      # FreeFriis SNR
        self.ObsDict["ConnUAV_UAV"] = self.MinMaxNorm(self.ConnUAV_UAV, 0, 30)     # FreeFriis SNR
        self.ObsDict["DataRateUE_UAV"] = self.MinMaxNorm(self.DataRateUE_UAV, 0, 3e6)  # OkumuraHata DR 23264662, 258345145
        self.ObsDict['UAV_Available_Lst_onehot'] = self.UAV_Available_Lst_onehot.copy()
        self.ObsDict["UE_MaxDR"] = self.MinMaxNorm(self.UE_MaxDR, 0, 3e7)    # OkumuraHata DR 23264662, 258345145
        self.ObsDict["UE_UAVid"] = (np.array(self.UE_UAVid)+1) / env.NUM_UAV        # 0 - #uav数目

        UAV_Locs = []
        for i in env.UAV.values():
            UAV_Locs.append([i.x, i.y, i.height])
        UE_Locs = []
        for i in env.UE.values():
            UE_Locs.append([i.x, i.y])
        BS_Locs = []
        for i in env.BS.values():
            BS_Locs.append([i.x, i.y])
        normFactor = max(env.width, env.height, env.h_3d)
        self.ObsDict['Locs_UAV'] = np.array(UAV_Locs) / normFactor
        self.ObsDict['Locs_UE'] = np.array(UE_Locs) / normFactor
        self.ObsDict['Locs_BS'] = np.array(BS_Locs) / normFactor
        self.normFactor = normFactor

        return

    def MinMaxNorm(self, l, min, max):
        l = np.clip(l, min, max)
        # print(round(np.max((l - min) / (max - min)), 6))
        return (l - min) / (max - min)