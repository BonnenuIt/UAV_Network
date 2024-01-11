import os
from base_marl_ray_fcoop import myCommEnvMA_FCOOP
from marllib import marl
from base_marl_ray import myCommEnvMA
from marl_train import my_ippo, my_mappo, my_qmix
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,1,4,3,5,2"
    ENV_REGISTRY["my_comm"] = myCommEnvMA
    COOP_ENV_REGISTRY["my_comm"] = myCommEnvMA_FCOOP

    # initialize env
    env = marl.make_env(environment_name="my_comm", 
                        map_name="myCommEnvMA", 
                        abs_path="my_comm.yaml")
    # customize model
    mappo = my_ippo()
    # mappo = my_mappo()
    # mappo = my_maa2c()
    # mappo = my_matrpo()
    # mappo = my_qmix()
    # mappo = my_vdn()
    # mappo = my_vdppo()
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "512-256-128-256-256"})

    # rendering
    mappo.render(
        env, model, 
        stop={'episode_reward_mean': 200000, 'timesteps_total': 1000}, 
        local_mode=True,   # True for debug mode only
        share_policy='group',   # individual(separate) / group(division) / all(share)
        num_gpus=1,
        num_workers=1,
        # checkpoint_freq=50,   # save model every X training iterations
        restore_path={'params_path': "exp_results_new/ippo_mlp_myCommEnvMA/IPPOTrainer_my_comm_myCommEnvMA_a5e41_00000_0_2023-12-23_23-05-02/params.json",
                    'model_path': "exp_results_new/ippo_mlp_myCommEnvMA/IPPOTrainer_my_comm_myCommEnvMA_a5e41_00000_0_2023-12-23_23-05-02/checkpoint_004500/checkpoint-4500", 
                    'render':True,
                    }
    )
    # mappo.fit(env, model, 
    #           stop={'episode_reward_mean': 2000000, 'timesteps_total': 200000000}, 
    #           local_mode=True,     # True for debug mode only
    #           share_policy='group',   # individual(separate) / group(division) / all(share)
    #           num_gpus=1,
    #           num_workers=3, 
    #           checkpoint_freq=50,   # save model every X training iterations
    #           )

# conda activate marllib
# python marl_render.py
# python my_animation.py