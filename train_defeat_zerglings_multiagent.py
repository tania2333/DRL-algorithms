import sys
import os
import datetime
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold = 1e6)

from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions
from BicNet_segment_training import learn

from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
import baselines.common.tf_util as U
from BicNet import train_bicnet as tb

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
  steps_left = FLAGS.timesteps

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

  lr = FLAGS.lr
  buffer_size = 50000
  batch_size = 32  # 32
  gamma = 1.0
  num_agents = 9
  vector_obs_len = 4096  # 32*32  1024
  output_len = 1
  hidden_vector_len = 1
  tau = 0.001


  sess = U.make_session()
  sess.__enter__()
  actor = tb.ActorNetwork(sess, lr, tau, batch_size, num_agents, vector_obs_len, output_len, hidden_vector_len)
  critic = tb.CriticNetwork(sess, lr, tau, gamma, actor.get_num_trainable_vars(), num_agents, vector_obs_len,
                            output_len, hidden_vector_len)
  replay_buffer = tb.ReplayBuffer(buffer_size)

  while(steps_left > 0):
    with sc2_env.SC2Env(
        map_name="DefeatZerglingsAndBanelings",  #DefeatZerglingsAndBanelings
        step_mul=step_mul,
        agent_interface_format=AGENT_INTERFACE_FORMAT,
        visualize=True, #True
        game_steps_per_episode=steps * step_mul) as env:

      steps_left = learn(
        env,
        sess=sess,
        lr=FLAGS.lr,
        max_timesteps=FLAGS.timesteps,
        steps_left=steps_left,
        train_freq=1,
        learning_starts=1000,#100000,
        target_network_update_freq=20,#1000
        gamma=0.99,
        callback=BicNet_callback,
        actor=actor,
        critic=critic,
        replay_buffer=replay_buffer,
        num_agents = num_agents
      )
#
def BicNet_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if('flag_end' in locals and locals['flag_end'] == True):
    if('mean_100ep_reward' in locals
       and locals['num_episodes'] >= 10  #100
       and locals['mean_100ep_reward'] > max_mean_reward
       ):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      max_mean_reward = locals['mean_100ep_reward']
#
#
if __name__ == '__main__':
  main()
#
#
#
