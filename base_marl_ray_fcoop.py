from base_marl_ray import myCommEnvMA

class myCommEnvMA_FCOOP(myCommEnvMA):
    def step(self, actions):
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
        rewards = self.handler.reward_fcoop(self)

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

        # info = self.myLink.ObsDict
        # info['normFactor'] = self.myLink.normFactor
        info = {}

        # check whether episode is done & close the environment
        if self.time >= self.EP_MAX_TIME:
            self.closed = True
        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        # terminated = self.closed
        terminated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}
        terminated["__all__"] = self.closed
        truncated = {str(uav.uav_id): self.closed for uav in self.UAV.values()}

        return obs, rewards, terminated, info

    