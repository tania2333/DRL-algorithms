import sys
import os

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions
import os

import deepq_mineral_shards
import datetime

from common.vec_env.subproc_vec_env import SubprocVecEnv
from a2c.policies import CnnPolicy
from a2c import a2c
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import random

import deepq_mineral_4way
import deepq_ActSeparate
import deepq_CoordAndActAsOutput
import deepq_actSeparateWith4Directions
import deepq_actionGroup_4way
import deep_DiffActInSameTime

import threading
import time

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8 #8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "deepq-4way", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train") #2000000
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")
flags.DEFINE_integer("num_agents", 4, "number of RL agents for A2C")
flags.DEFINE_integer("num_scripts", 0, "number of script agents for A2C")
flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%m%d%H%M")


def main():
  FLAGS(sys.argv)

  print("algorithm : %s" % FLAGS.algorithm)
  print("timesteps : %s" % FLAGS.timesteps)
  print("exploration_fraction : %s" % FLAGS.exploration_fraction)
  print("prioritized : %s" % FLAGS.prioritized)
  print("dueling : %s" % FLAGS.dueling)
  print("num_agents : %s" % FLAGS.num_agents)
  print("lr : %s" % FLAGS.lr)

  if (FLAGS.lr == 0):
    FLAGS.lr = random.uniform(0.00001, 0.001)

  print("random lr : %s" % FLAGS.lr)

  lr_round = round(FLAGS.lr, 8)

  logdir = "tensorboard"

  if (FLAGS.algorithm == "deepq-4way"):
    logdir = "tensorboard/mineral/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
      FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)
  elif (FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/mineral/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
      FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)
  elif (FLAGS.algorithm == "a2c"):
    logdir = "tensorboard/mineral/%s/%s_n%s_s%s_nsteps%s/lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps,
      FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts,
      FLAGS.nsteps, lr_round, start_time)

  if (FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif (FLAGS.log == "stdout"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(sys.stdout)])

  if (FLAGS.algorithm == "deepq"):
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(         #interface.feature_layer.resolution 和  interface.feature_layer.minimap_resolution
      feature_dimensions=sc2_env.Dimensions(screen=32, minimap=32)   # 16 16
      # feature_dimensions = sc2_env.Dimensions(screen=32, minimap=32)  # 16 16
    )
    with sc2_env.SC2Env(
        map_name="CollectMineralShards",
        step_mul=step_mul,   #推进的速度，通俗理解就是人类玩家的每秒的有效操作
        visualize=True,
        # screen_size_px=(16, 16),
        # minimap_size_px=(16, 16)) as env:
        agent_interface_format=AGENT_INTERFACE_FORMAT) as env:

      model = deepq.models.cnn_to_mlp(  #his model takes as input an observation and returns values of all actions.注意如何在deepq_mineral_shards.learn用到该model
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)  #卷积核数量,卷积核大小,步长
        # convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[512], dueling=True)  # 卷积核数量,卷积核大小,步长
      act = deepq_mineral_shards.learn(  #训练模型并保存
      # act = deepq_ActSeparate.learn(  #训练模型并保存
      # act=deepq_actSeparateWith4Directions.learn(
      # act = deepq_actionGroup_4way.learn(
      # act = deep_DiffActInSameTime.learn(
        env,
        q_func=model,
        num_actions=4,   #default 16  num_actions=256   3  4
        lr=FLAGS.lr,
        max_timesteps=FLAGS.timesteps,
        buffer_size=10000,
        exploration_fraction=FLAGS.exploration_fraction,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=deepq_actSeparateWith4Directions_callback) #deepq_callback; deepq_ActSeperate_callback  ;   deepq_actSeparateWith4Directions_callback  deep_DiffActInSameTime_callback
      act.save("mineral_shards.pkl")   #在所有训练步骤之后将训练过的模型保存到mineral_shards.pkl文件中, 用于enjoy_mineral_shards.py

  elif (FLAGS.algorithm == "deepq-4way"):
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(   
      feature_dimensions=sc2_env.Dimensions(screen=32, minimap=32)
    )
    with sc2_env.SC2Env(  #
        map_name="CollectMineralShards",
        step_mul=step_mul,
        # screen_size_px=(32, 32),
        # minimap_size_px=(32, 32),
        save_replay_episodes=2,
        replay_dir="D:/StarCraft II/StarCraft II/video",
        agent_interface_format=AGENT_INTERFACE_FORMAT,
        visualize=True) as env:

      model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
      # model = deepq.models.mlp(hiddens=[256,128,4])
      act = deepq_mineral_4way.learn(
        env,
        q_func=model,
        num_actions=4,
        lr=FLAGS.lr,
        max_timesteps=FLAGS.timesteps,
        buffer_size=10000,
        exploration_fraction=FLAGS.exploration_fraction,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=deepq_4way_callback)

      act.save("mineral_shards.pkl")

  elif (FLAGS.algorithm == "a2c"):

    num_timesteps = int(40e6)

    num_timesteps //= 4

    seed = 0

    env = SubprocVecEnv(FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts, FLAGS.map)

    policy_fn = CnnPolicy
    a2c.learn(
      policy_fn,
      env,
      seed,
      total_timesteps=num_timesteps,
      nprocs=FLAGS.num_agents + FLAGS.num_scripts,
      nscripts=FLAGS.num_scripts,
      ent_coef=0.5,
      nsteps=FLAGS.nsteps,
      max_grad_norm=0.01,
      callback=a2c_callback)


from pysc2.env import environment
import numpy as np
def deepq_actionGroup_4way_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(
          os.path.join(PROJ_DIR, 'models/deepq_actionGroup_4way/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq_actionGroup_4way/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act = deepq_mineral_4way.ActWrapper(locals['act'])
      #act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(PROJ_DIR,
                              'models/deepq_actionGroup_4way/mineral_%s.pkl' %
                              locals['mean_100ep_reward'])
      act.save(filename)
      # filename = os.path.join(
      #   PROJ_DIR,
      #   'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      # act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename


def deepq_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10  #should be mean_10ep_reward rather than 100
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
      act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
      act_x.save(filename)
      filename = os.path.join(
        PROJ_DIR,
        'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename

def deepq_ActSeperate_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq_actSeparate/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq_actSeparate/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
      act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparate/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
      act_x.save(filename)
      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparate/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename

def deepq_4way_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(
          os.path.join(PROJ_DIR, 'models/deepq-4way/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq-4way/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act = deepq_mineral_4way.ActWrapper(locals['act'])
      #act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(PROJ_DIR,
                              'models/deepq-4way/mineral_%s.pkl' %
                              locals['mean_100ep_reward'])
      act.save(filename)
      # filename = os.path.join(
      #   PROJ_DIR,
      #   'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      # act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename

def deepq_actSeparateWith4Directions_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10  #should be mean_10ep_reward rather than 100
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq_actSeparateWith4Directions/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq_actSeparateWith4Directions/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
      act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparateWith4Directions/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
      act_x.save(filename)
      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparateWith4Directions/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename


def deep_DiffActInSameTime_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10  #should be mean_10ep_reward rather than 100
        and locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq_actSeparateWith4Directions/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq_actSeparateWith4Directions/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
      act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparateWith4Directions/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
      act_x.save(filename)
      filename = os.path.join(
        PROJ_DIR,
        'models/deepq_actSeparateWith4Directions/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename




def a2c_callback(locals, globals):
  global max_mean_reward, last_filename
  #pprint.pprint(locals)

  if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10
      and locals['mean_100ep_reward'] > max_mean_reward):
    print("mean_100ep_reward : %s max_mean_reward : %s" %
          (locals['mean_100ep_reward'], max_mean_reward))

    if (not os.path.exists(os.path.join(PROJ_DIR, 'models/a2c/'))):
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/'))
      except Exception as e:
        print(str(e))
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/a2c/'))
      except Exception as e:
        print(str(e))

    if (last_filename != ""):
      os.remove(last_filename)
      print("delete last model file : %s" % last_filename)

    max_mean_reward = locals['mean_100ep_reward']
    model = locals['model']

    filename = os.path.join(
      PROJ_DIR,
      'models/a2c/mineral_%s.pkl' % locals['mean_100ep_reward'])
    model.save(filename)
    print("save best mean_100ep_reward model to %s" % filename)
    last_filename = filename


if __name__ == '__main__':
  main()
