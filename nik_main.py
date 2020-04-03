import sys
import re
import multiprocessing
import os.path as osp
import gym
import pybulletgym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def train(config, log_file):
    alg_config = config
    # print('\n\n............running config...........'+str(exp_number))
    # print(alg_config)
    # print('\n\n')
    # arg_parser = common_arg_parser()
    # args, unknown_args = arg_parser.parse_known_args(args)
    # extra_args = parse_cmdline_kwargs(unknown_args)
    log_path = log_file
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(log_path, format_strs=[])


    env_type, env_id, alg, seed, total_timesteps, network, gpu_device = alg_config
    print('env_type: {}'.format(env_type))

    learn = get_learn_function(alg)
    alg_kwargs = dict(
        nsteps=1000,
        value_network='copy',
        network=network,
        lr=0.25,
        lrschedule='linear'
    )
    # alg_kwargs.update(extra_args)

    env = build_env(alg, seed, env_type, env_id)
    # if args.save_video_interval != 0:
    #     env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    print('...............................................')
    print(alg_kwargs)
    print('Training {} on {}:{} with arguments \n{}'.format(alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        #gpu_device=gpu_device,
        **alg_kwargs
    )
    return model, env

def main(main_config):
    # env_type ='mujoco', env='HalfCheetahMuJoCoEnv-v0', alg='acktr', seed=22, total_timesteps=1000000, network='mlp', gpu_device=1, log_file='./test/'
    env_type, env, alg, seed, total_timesteps, network, gpu_device, log_file = main_config
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
     # try:
      #  tf.config.experimental.set_virtual_device_configuration(
       #     gpus[int(gpu_device)],
       #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7240)])
       # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      #except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
      #  print(e)

    train([env_type, env, alg, int(seed), int(total_timesteps), network, gpu_device], log_file)
    # cheetah = ['mujoco', 'HalfCheetahMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # ant = ['mujoco', 'AntMuJoCotEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # hopper = ['mujoco', 'HopperMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # humanoid = ['mujoco', 'HumanoidMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # walker = ['mujoco', 'Walker2DMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # configs = [cheetah, ant, hopper, humanoid, walker]
            # configure logger, disable logging in child MPI processes (with rank > 0)




def build_env(alg, seed, env_type, env_id):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = 1 or ncpu
    alg = alg
    seed = seed

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=None, reward_scale=1.0)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, 1, seed, 1.0, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

if __name__ == '__main__':
    main(sys.argv[1].split(','))

