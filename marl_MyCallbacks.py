import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index: int, **kwargs):
        episode.custom_metrics["NUM_UE_CONN"] = base_env.envs[0].NUM_UE_CONN
        episode.custom_metrics["UE_DR_TOTAL"] = base_env.envs[0].UE_DR_TOTAL
        for i in base_env.envs:
            i.curriculum += worker.num_workers * base_env.envs[0].EP_MAX_TIME
        # envs = base_env.get_unwrapped()
        # workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_task(task)))

        
        # base_env.envs[0].UE_DR_TOTAL
        # episode.custom_metrics["pole_angle"] = 1
        # # Make sure this episode is really done.
        # return
        # assert episode.batch_builder.policy_collectors[
        #     "default_policy"].batches[-1]["dones"][-1], \
        #     "ERROR: `on_episode_end()` should only be called " \
        #     "after episode is done!"
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print("episode {} (env-idx={}) ended with length {} and pole "
        #       "angles {}".format(episode.episode_id, env_index, episode.length,
        #                          pole_angle))
        # episode.custom_metrics["pole_angle"] = pole_angle
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
    # def on_train_result(self, algorithm, result, **kwargs):
    #     pass
