import sys
import os
import datetime
import tensorflow as tf

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from defeat_zerglings import dqfd
from BicNet import BicNet_findAndDefeatZergling
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8 #1
steps = 2000

FLAGS = flags.FLAGS
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "BicNet", "RL algorithm to use.")  #deepq  BicNet
flags.DEFINE_integer("timesteps", 1000000, "Steps to train")  #2000000
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_float("num_cpu", 16, "Number of CPU")


PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")


def main():

  # tf.reset_default_graph()
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True

  FLAGS(sys.argv)

  logdir = "tensorboard"
  if(FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.exploration_fraction,
      FLAGS.prioritized,
      FLAGS.dueling,
      FLAGS.lr,
      start_time
    )
  elif(FLAGS.algorithm == "acktr"):
    logdir = "tensorboard/zergling/%s/%s_num%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.num_cpu,
      FLAGS.lr,
      start_time
    )
  elif(FLAGS.algorithm == "BicNet"):
    logdir = "tensorboard/zergling/%s/%s_num%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.num_cpu,
      FLAGS.lr,
      start_time
    )

  if(FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif(FLAGS.log == "stdout"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(sys.stdout)])

  AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64)#feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64)  将他俩处理成32*32的矩阵
  )
  with sc2_env.SC2Env(
      map_name="DefeatZerglingsAndBanelings",  #DefeatZerglingsAndBanelings
      step_mul=step_mul,
      agent_interface_format=AGENT_INTERFACE_FORMAT,
      visualize=True, #True
      game_steps_per_episode=steps * step_mul) as env:

    model = deepq.models.cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
      hiddens=[256],
      dueling=True
    )
    #model,需要改成lstm的形式。
    demo_replay = []
    # act = dqfd.learn(
    #   env,
    #   q_func=model,
    #   num_actions=3,
    #   lr=1e-4,
    #   max_timesteps=10000000,
    #   buffer_size=100000,
    #   exploration_fraction=0.5,
    #   exploration_final_eps=0.01,
    #   train_freq=2,
    #   learning_starts=100000,
    #   target_network_update_freq=1000,
    #   gamma=0.99,
    #   prioritized_replay=True,
    #   callback=deepq_callback
    # )
    # act.save("defeat_zerglings.pkl")
    BicNet_findAndDefeatZergling.learn(
      env,
      lr=FLAGS.lr,
      max_timesteps=FLAGS.timesteps,
      buffer_size=100000,
      train_freq=1,
      learning_starts=1000,#100000,
      target_network_update_freq=1000,
      gamma=0.99,
      callback=BicNet_callback
    )

def deepq_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if('done' in locals and locals['done'] == True):
    if('mean_100ep_reward' in locals
       and locals['num_episodes'] >= 10
       and locals['mean_100ep_reward'] > max_mean_reward
       ):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if(not os.path.exists(os.path.join(PROJ_DIR,'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/deepq/'))
        except Exception as e:
          print(str(e))

      if(last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act = dqfd.ActWrapper(locals['act'])

      filename = os.path.join(PROJ_DIR,'models/deepq/zergling_%s.pkl' % locals['mean_100ep_reward'])
      act.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename

def acktr_callback(locals, globals):
  global max_mean_reward, last_filename
  #pprint.pprint(locals)

  if('mean_100ep_reward' in locals
     and locals['num_episodes'] >= 10
     and locals['mean_100ep_reward'] > max_mean_reward
     ):
    print("mean_100ep_reward : %s max_mean_reward : %s" %
          (locals['mean_100ep_reward'], max_mean_reward))

    if(not os.path.exists(os.path.join(PROJ_DIR,'models/acktr/'))):
      try:
        os.mkdir(os.path.join(PROJ_DIR,'models/'))
      except Exception as e:
        print(str(e))
      try:
        os.mkdir(os.path.join(PROJ_DIR,'models/acktr/'))
      except Exception as e:
        print(str(e))

    if(last_filename != ""):
      os.remove(last_filename)
      print("delete last model file : %s" % last_filename)

    max_mean_reward = locals['mean_100ep_reward']
    model = locals['model']

    filename = os.path.join(PROJ_DIR,'models/acktr/zergling_%s.pkl' % locals['mean_100ep_reward'])
    model.save(filename)
    print("save best mean_100ep_reward model to %s" % filename)
    last_filename = filename

def BicNet_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if('flag_end' in locals and locals['flag_end'] == True):
    if('mean_100ep_reward' in locals
       and locals['num_episodes'] >= 100
       and locals['mean_100ep_reward'] > max_mean_reward
       ):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))
      max_mean_reward = locals['mean_100ep_reward']


if __name__ == '__main__':
  main()


