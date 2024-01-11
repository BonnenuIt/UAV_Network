import os
import time
from base_marl_ray_fcoop import myCommEnvMA_FCOOP
from marllib import marl
from base_marl_ray import myCommEnvMA
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

def my_ippo():
    print("ippo")
    return marl.algos.ippo(hyperparam_source="common", batch_episode=8)

def my_vdppo():
    print("vdppo")
    return marl.algos.vdppo(hyperparam_source="common", batch_episode=8)

def my_vdn():
    print("vdn")
    return marl.algos.vdn(hyperparam_source="common", batch_episode=8)

def my_qmix():
    print("qmix")
    return marl.algos.qmix(hyperparam_source="common", batch_episode=8)

def my_maa2c():
    print("maa2c")
    return marl.algos.maa2c(hyperparam_source="common", batch_episode=8)

def my_matrpo():
    print("matrpo")
    return marl.algos.matrpo(hyperparam_source="common", batch_episode=8)

def my_mappo():
    print("mappo")
    return marl.algos.mappo(hyperparam_source="common", batch_episode=8)

def m_main(mappo, my_share_policy):
    # initialize env
    env = marl.make_env(environment_name="my_comm", 
                        map_name="myCommEnvMA", 
                        abs_path="my_comm.yaml")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "512-256-128-256-256"})
    
    # start learning
    mappo.fit(env, model, 
              stop={'episode_reward_mean': 2000000, 'timesteps_total': 100000000}, 
              local_mode=False,     # True for debug mode only
              share_policy=my_share_policy,   # individual(separate) / group(division) / all(share)
              num_gpus=1,
              num_workers=8, 
              checkpoint_freq=300,   # save model every X training iterations
            #   restore_path={'params_path': "exp_results_new/mappo_mlp_myCommEnvMA/MAPPOTrainer_my_comm_myCommEnvMA_102cb_00000_0_2023-12-08_18-46-37/params.json",
            #             'model_path': "exp_results_new/mappo_mlp_myCommEnvMA/MAPPOTrainer_my_comm_myCommEnvMA_102cb_00000_0_2023-12-08_18-46-37/checkpoint_002450/checkpoint-2450", 
            #         }
              )

if __name__ == '__main__':
    RL_policy = "ippo"  # ippo  mappo
    my_share_policy = 'group'   # # individual(separate) / group(division) / all(share)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,5,4,1,2,3"
    
    '''
    nohup python marl_train.py > logs/ippo_group27.log 2>&1 &

    nohup python marl_train.py --overwrite --output_dir logs_new/debug --gpu --gpu_id 7 > logs_new/vdn.log 2>&1 &
    ps -ef | grep marl_train 然后可以查看后台运行的东西, ps -u xyg22   ps aux | grep 78240     ps -f -p 78240 
    ps -p 79491 -o lstart 启动时间 
    
    nvidia-smi | grep ray
    kill -9 xxxxx 进程号可以终止，后台的输出可以在 .log里面看
    conda config --get channels
    conda info
    tensorboard --logdir=exp_results_new/ippo_mlp_myCommEnvMA --port=6001
    tensorboard --logdir=exp_results_new/ippo_mlp_myCommEnvMA --samples_per_plugin scalars=999999 --port=6002
    tensorboard --logdir=/home/xyg22/Network/my-v2/MARL/exp_results
    conda activate marllib
    tensorboard --logdir=exp_results_new/ippo_mlp_myCommEnvMA/IPPOTrainer_my_comm_myCommEnvMA_a2677_00000_0_2023-12-20_16-49-06 --port=6001
    '''

    print("当前进程：", os.getpid(), " 父进程：", os.getppid())
    print("Time =", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # register new env
    ENV_REGISTRY["my_comm"] = myCommEnvMA
    COOP_ENV_REGISTRY["my_comm"] = myCommEnvMA_FCOOP
    # 
    # mappo = my_maa2c()
    # mappo = my_matrpo()
    # mappo = my_qmix()
    # mappo = my_vdn()
    # mappo = my_vdppo()
    if RL_policy == "ippo":
        mappo = my_ippo()
    elif RL_policy == "mappo":
        mappo = my_mappo()
    
    print("Policy:", my_share_policy)
    m_main(mappo, my_share_policy)

    print("END")
    # # rendering
    # mappo.render(
    # env, model, 
    # local_mode=True, 
    # restore_path={'params_path': "checkpoint/params.json",
    #                 'model_path': "checkpoint/checkpoint-10"}
    # )
