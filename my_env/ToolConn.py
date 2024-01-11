import copy
import numpy as np
import time

class ToolConn():
    def __init__(self, snr_bs_th, snr_uav_th):
        self.snr_bs_th = snr_bs_th
        self.snr_uav_th = snr_uav_th

    def reset(self, myTools, env):
        self.myTools = myTools
        self.UAV_Available()
        self.UE_DataRate()
        self.Obs_normalise(env)

    def UAV_Available(self):    # myTools.O_ConnBS_UAV - (BS, UAV)
        NUM_UAV = self.myTools.O_ConnUAV_UAV.shape[0]

        BSid = np.max(self.myTools.O_ConnBS_UAV, axis=0)
        C_BS_UAV_UAVid = np.where(BSid >= self.snr_bs_th)[0]
        if C_BS_UAV_UAVid.shape[0] == 0:
            self.Link_BS = [-1] * NUM_UAV
            self.Link_backward = [-1] * NUM_UAV
            self.Link_forward = [[]] * NUM_UAV
        C_BS_UAV_BSid = np.argmax(self.myTools.O_ConnBS_UAV[:, C_BS_UAV_UAVid], axis=0)

        UAV_Available_Lst = C_BS_UAV_UAVid.tolist()
        linkQuality = [-np.inf] * NUM_UAV
        C_BS_Lst = [-1] * NUM_UAV
        backLinks = [-1] * NUM_UAV
        forwardLinks = [[]] * NUM_UAV
        for i in range(C_BS_UAV_UAVid.shape[0]):
            j = C_BS_UAV_UAVid[i]
            linkQuality[j] = np.inf
            C_BS_Lst[j] = C_BS_UAV_BSid[i]
            backLinks[j] = []

        i = 0
        while(True):
            if i == len(UAV_Available_Lst):
                break
            UAVid = UAV_Available_Lst[i]    # 看UAVid连上的无人机
            Base_linkQuality = (NUM_UAV - len(backLinks[UAVid])) * 1e6    # 越大 linkQuality越好(加负号) (14-跳数)*1e6

            ConnUAV_i = self.myTools.O_ConnUAV_UAV[UAVid, :]
            C_UAV_UAVid = np.where(ConnUAV_i >= self.snr_uav_th)[0]

            l_last = []
            for k in C_UAV_UAVid:
                if (k not in C_BS_UAV_UAVid) and k != UAVid:
                    l_last += [k]
            C_UAV_UAVid = np.array(l_last)

            # C_UAV_UAVid = np.setdiff1d(C_UAV_UAVid, C_BS_UAV_UAVid)     # 去掉直连基站的 差集
            # C_UAV_UAVid = np.setdiff1d(C_UAV_UAVid, np.array([UAVid]))  # 去掉自己 剩下真正连上的

            for j in C_UAV_UAVid:   # 遍历连上的
                j_linkQuality = ConnUAV_i[j] + Base_linkQuality # 越大 linkQuality越好  SNR + (14-跳数)*1e6
                if j_linkQuality >= linkQuality[j]:
                    linkQuality[j] = copy.deepcopy(j_linkQuality)
                    backLinks[j] = backLinks[UAVid] + [UAVid]
                    if j not in UAV_Available_Lst:
                        UAV_Available_Lst.append(j)
            i += 1

        for i in range(len(backLinks)): # 制作forwardLinks
            if isinstance(backLinks[i], list) and len(backLinks[i]) > 0:
                for j in backLinks[i]:
                    forwardLinks[j] = forwardLinks[j] + [i]

        self.Link_BS = C_BS_Lst
        self.Link_backward = backLinks
        self.Link_forward = forwardLinks

        UAV_Available_Lst = sorted(list(set(UAV_Available_Lst)))    # 排序去重
        self.UAV_Available_Lst = UAV_Available_Lst
        self.UAV_Available_Onehot = self.List2OneHot1D(UAV_Available_Lst, NUM_UAV)
        return
    
    def List2OneHot1D(self, UAV_Available_Lst, NUM_UAV):
        res = np.zeros((NUM_UAV))
        for i in UAV_Available_Lst:
            res[i] += 1
        return res
    
    def UE_DataRate(self):
        if self.UAV_Available_Lst == []:
            UE_MaxDR = np.zeros(self.myTools.O_DataRateUE_UAV.shape[0])
            UE_UAVid = [-1] * self.myTools.O_DataRateUE_UAV.shape[0]

        else:
            UE_Conn = self.myTools.O_DataRateUE_UAV[:, self.UAV_Available_Lst]  # [ue, avai uav]
            UE_MaxDR = np.max(UE_Conn, axis=1)
            UE_UAVid = [-1] * self.myTools.O_DataRateUE_UAV.shape[0]

            C_UEid = np.where(UE_MaxDR > 0.001)[0]   # UAV 连接上的 ue的id   （防止DR近似0大于0）

            # UE_UAVid_fake每个UE连上的UAV的假id，实际是UAV_Available_Lst的索引
            UE_UAVid_fake = np.argmax(UE_Conn, axis=1)  # 去除后的索引值，不是对应的所有UAV，而只是可用UAV的索引
            for i in C_UEid:    # 连接上的 ue的id
                UE_UAVid[i] = self.UAV_Available_Lst[UE_UAVid_fake[i]]

        self.UE_MaxDR = UE_MaxDR
        self.UE_UAVid = UE_UAVid
        return
    
    def Obs_normalise(self, env):
        NUM_UAV = self.myTools.O_ConnUAV_UAV.shape[0]

        self.ObsDict = {}

        # print("**************")
        # print(np.max(self.myTools.O_ConnBS_UAV), np.min(self.myTools.O_ConnBS_UAV))
        # aa = np.where(self.myTools.O_ConnUAV_UAV > 1e5, 0, self.myTools.O_ConnUAV_UAV)
        # b = np.max(self.myTools.O_DataRateUE_UAV)
        # bb = np.where(self.myTools.O_DataRateUE_UAV < 1, b, self.myTools.O_DataRateUE_UAV)
        # print(np.max(aa), np.min(aa))
        # print(np.max(bb), np.min(bb))
        # print(np.max(self.UE_MaxDR), np.min(self.UE_MaxDR))

        self.ObsDict["ConnBS_UAV"] = self.MinMaxNorm(self.myTools.O_ConnBS_UAV, 0, 2000)      # FreeFriis SNR
        self.ObsDict["ConnUAV_UAV"] = self.MinMaxNorm(self.myTools.O_ConnUAV_UAV, 0, 7)     # FreeFriis SNR
        self.ObsDict["DataRateUE_UAV"] = self.MinMaxNorm(self.myTools.O_DataRateUE_UAV, 1e5, 2e7)  # OkumuraHata DR 23264662, 258345145
        self.ObsDict['UAV_Available_Lst_onehot'] = self.UAV_Available_Onehot.copy()
        self.ObsDict["UE_MaxDR"] = self.MinMaxNorm(self.UE_MaxDR, 1e5, 2e7)    # OkumuraHata DR 23264662, 258345145
        self.ObsDict["UE_UAVid"] = (np.array(self.UE_UAVid) + 1) / NUM_UAV        # 0 - #uav数目

        normFactor = max(env.width, env.height, env.h_3d)
        self.ObsDict['Locs_UAV'] = self.myTools.UAV_position / normFactor
        self.ObsDict['Locs_UE'] = self.myTools.UE_position[:, :2] / normFactor
        self.ObsDict['Locs_BS'] = self.myTools.BS_position[:, :2] / normFactor
        self.normFactor = normFactor

        return

    def MinMaxNorm(self, l, min, max):
        l = np.clip(l, min, max)
        # print(round(np.max((l - min) / (max - min)), 6))
        return (l - min) / (max - min)
            