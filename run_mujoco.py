#!/usr/bin/env python3

import sys
import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

def train(env_id, num_timesteps, seed):
    env = make_mujoco_env(env_id, seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.995, lam=0.97, timesteps_per_batch=1000,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

        env.close()

def main(main_config):
    # env_type ='mujoco', env='HalfCheetahMuJoCoEnv-v0', alg='acktr', seed=22, total_timesteps=1000000, network='mlp', gpu_device=1, log_file='./test/'
    env_type, env, alg, seed, total_timesteps, network, gpu_device, log_file = main_config
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #   try:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpus[int(gpu_device)],
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6240)])
    #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #   except RuntimeError as e:
    #     # Virtual devices must be set before GPUs have been initialized
    #     print(e)
    logger.configure(log_file)
    train(env, num_timesteps=int(total_timesteps), seed=int(seed))
    # train([env_type, env, alg, int(seed), int(total_timesteps), network, int(gpu_device)], log_file)
    # cheetah = ['mujoco', 'HalfCheetahMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # ant = ['mujoco', 'AntMuJoCotEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # hopper = ['mujoco', 'HopperMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # humanoid = ['mujoco', 'HumanoidMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]
    # walker = ['mujoco', 'Walker2DMuJoCoEnv-v0', 'acktr', 22, 10, 'mlp', 1]

# def main():
#     args = mujoco_arg_parser().parse_args()
#     logger.configure()
#     train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == "__main__":
    main(sys.argv[1].split(','))

