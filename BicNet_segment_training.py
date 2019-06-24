import numpy as np
np.set_printoptions(threshold = 1e6)

import os
import tempfile
import datetime
from absl import flags
import baselines.common.tf_util as U
from baselines import logger
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

from common import common_group


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

FLAGS = flags.FLAGS

def learn(env,
          sess,
          actor,
          critic,
          replay_buffer,
          lr=5e-4,
          max_timesteps=100000,
          segment_steps=10000,#10000
          steps_left=100000,
          train_freq=1,  #1
          batch_size=32,  #32
          print_freq=1,
          checkpoint_freq=100, #10000
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=20,            #500,
          #num_cpu=16,
          callback=None,
          num_agents=9,
          stddev=0.2,
          action_low=-1,
          action_high=1
          ):
#   """Train a deepq model.
#
# Parameters
# -------
# env: pysc2.env.SC2Env
#     environment to train on
# q_func: (tf.Variable, int, str, bool) -> tf.Variable
#     the model that takes the following inputs:
#         observation_in: object
#             the output of observation placeholder
#         num_actions: int
#             number of actions
#         scope: str
#         reuse: bool
#             should be passed to outer variable scope
#     and returns a tensor of shape (batch_size, num_actions) with values of every action.
# lr: float
#     learning rate for adam optimizer
# max_timesteps: int
#     number of env steps to optimizer for
# buffer_size: int
#     size of the replay buffer
# train_freq: int
#     update the model every `train_freq` steps.
#     set to None to disable printing
# batch_size: int
#     size of a batched sampled from replay buffer for training
# print_freq: int
#     how often to print out training progress
#     set to None to disable printing
# checkpoint_freq: int
#     how often to save the model. This is so that the best version is restored
#     at the end of the training. If you do not wish to restore the best version at
#     the end of the training set this variable to None.
# learning_starts: int
#     how many steps of the model to collect transitions for before learning starts
# gamma: float
#     discount factor
# target_network_update_freq: int
#     update the target network every `target_network_update_freq` steps.
# num_cpu: int
#     number of cpus to use for training
# callback: (locals, globals) -> None
#     function called at every steps with state of the algorithm.
#     If callback returns true training stops.
#
# Returns
# -------
# act: ActWrapper
#     Wrapper over act function. Adds ability to save it and load it.
#     See
#
#
#     of baselines/deepq/categorical.py for details on the act function.
# """
#   # Create all the functions necessary to train the model
#
  # tf.reset_default_graph()
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # sess = tf.Session(config = config)
  # sess.__enter__()
  doc_name = max_timesteps - steps_left
  previous_doc_name = doc_name - segment_steps
  model_file_load = os.path.join(str(previous_doc_name)+"_"+"model_segment_training/", "defeat_zerglings")
  model_file_save = os.path.join(str(doc_name)+"_"+"model_segment_training/", "defeat_zerglings")

  obs = env.reset()
  obs = common_group.init(env, obs)
  action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=float(stddev) * np.ones(1))

  episode_rewards = [0.0]
  saved_mean_reward = None

  if (steps_left < max_timesteps):
    U.load_state(model_file_load, sess)
    learning_starts = 0
  else:
    U.initialize()
    # init = tf.global_variables_initializer()
    # sess.run(init)

  # actor.update_target_network()
  # critic.update_target_network()

  screen= obs[0].observation["feature_screen"][_UNIT_TYPE]

  group_id = 0
  count_episode = 0

  with tempfile.TemporaryDirectory() as td:
    model_saved = False

    for t in range(max_timesteps):
      startTime = datetime.datetime.now()
      if callback is not None:

        if callback(locals(), globals()):
          break

      # rew = np.zeros([9, 1]) #rew = 0
      new_action = None

      screen_expand = screenConcat(screen,num_agents)
      screen_input = np.expand_dims(screen_expand, axis=0)   # (1,3,1024)
      action = actor.predict(screen_input)[0]             #(1,3,1)->(3,1)
      act_with_noise = add_noise(action, action_low, action_high, action_noise)
      act = discret_act_func(act_with_noise)
      # print("act_discret",act)
      flag_end = False
      action_group = []
      group_list = common_group.update_group_list(obs)
      if (common_group.check_group_list(env, obs)):
        obs = common_group.init(env, obs)
        group_list = common_group.update_group_list(obs)
      unit_list = common_group.unit_position(obs, 1)
      for i in range(len(group_list)):
        group_id = group_list[i]
        obs, screen, player, Remove = common_group.select_marine(env, obs, group_id)
        if(Remove):
          continue
        if(group_id < act.shape[0]):
          obs, new_action = common_group.marine_action(env, obs, player, act[group_id])
        else:
          act_last_agent = 1
          obs, new_action = common_group.marine_action(env, obs, player, act_last_agent)
        army_count = env._obs[0].observation.player_common.army_count
        try:
            if army_count > 0 and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
              obs = env.step_rewrite(actions=new_action)
            else:
              new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
              obs = env.step(actions=new_action)
        except Exception as e:
          print(e)
          new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
          obs = env.step(actions=new_action)
      new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
      obs = env.step(actions=new_action)
      done_final = obs[0].step_type == environment.StepType.LAST
      rew = obs[0].reward
      # rew_per_unit = math.ceil(rew/len(group_list))
      rew_expand = np.zeros((num_agents, 1))
      for i in range(len(group_list)):
        group_id = group_list[i]
        if(group_id < num_agents):
          rew_expand[group_id] = rew #rew_per_unit
      if(done_final):
        flag_end = True
      # if(len(common_group.update_group_list(obs))==0):
      #   flag_end = True
      player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
      friend_y, friend_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if (len(friend_x) == 0):
        flag_end = True
      # selected = obs[0].observation["feature_screen"][_SELECTED]
      # player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
      new_screen = obs[0].observation["feature_screen"][_UNIT_TYPE]
      new_screen_expand = screenConcat(new_screen, num_agents)
      # Store transition in the replay buffer.
      replay_buffer.add(screen_expand, act_with_noise, rew_expand, new_screen_expand, flag_end) #[group0:[num_traces, trace.dimension], group1, ... group8]
      screen = new_screen

      episode_rewards[-1] += rew #rew.sum(axis=0)
      reward = episode_rewards[-1]
      if flag_end:
        count_episode += 1
        print("Episode Reward : %s" % episode_rewards[-1])
        obs = env.reset()
        obs = common_group.init(env, obs)
        screen = obs[0].observation["feature_screen"][
          _UNIT_TYPE]
        print('num_episodes is', len(episode_rewards))
        episode_rewards.append(0.0)
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=float(stddev) * np.ones(1))


        if t > learning_starts  and count_episode % train_freq == 0:      #t % train_freq == 0:
          print("training started")
          # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
          s_batch, a_batch, r_batch, s2_batch, done_batch = replay_buffer.sample_batch(batch_size)     #[group0:[batch_size, trace.dimension], group1, ... group8]
          target_q = r_batch + gamma * critic.predict_target(s2_batch, actor.predict_target(s2_batch))
          print("------------------")
          print("reward is")
          print(r_batch)
          print("------------------")
          print("Q'(a+1) is ")
          print(critic.predict_target(s2_batch, actor.predict_target(s2_batch)))
          print("------------------")
          predicted_q_value, _ = critic.train(s_batch, a_batch,
                                              np.reshape(target_q, (batch_size, num_agents, 1)))
          a_outs = actor.predict(s_batch)  # a_outs和a_batch是完全相同的
          grads = critic.action_gradients(s_batch, a_outs)  # delta Q对a的导数
          actor.train(s_batch, grads)  # 这里会计算a对θ的导数和最后的梯度

        if t > learning_starts and count_episode % target_network_update_freq == 0:#t % target_network_update_freq == 0:
          # Update target network periodically.
          actor.update_target_network()
          critic.update_target_network()

      mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
      num_episodes = len(episode_rewards)
      if flag_end and print_freq is not None and len(
          episode_rewards) % print_freq == 0:
        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", num_episodes)
        logger.record_tabular("reward", reward)
        logger.record_tabular("mean 100 episode reward",
                              mean_100ep_reward)
        logger.dump_tabular()

      if (checkpoint_freq is not None and t > learning_starts
          and num_episodes > 100 and t % checkpoint_freq == 0):
        if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
          if print_freq is not None:
            logger.log(
              "Saving model due to mean reward increase: {} -> {}".
                format(saved_mean_reward, mean_100ep_reward))
          # U.save_state(model_file_save)
          # model_saved = True
          saved_mean_reward = mean_100ep_reward

      endTime = datetime.datetime.now()
      time_used = str(endTime - startTime)
      print("t = %d, time used = %s" % (t,time_used))

      if(t>=segment_steps and steps_left >= segment_steps):
        U.save_state(model_file_save)
        model_saved = True
        steps_left -= segment_steps
        break

      elif(steps_left < segment_steps and t == steps_left):
        U.save_state(model_file_save)
        model_saved = True
        steps_left = 0
        break

    # if model_saved:
    #   if print_freq is not None:
    #     logger.log("Restored model with mean reward: {}".format(
    #       saved_mean_reward))
    #   U.load_state(model_file)

  # sess.close()
  return steps_left


def add_noise_per_unit(action, action_low, action_high, noise):
  # s = np.random.normal(0, 0.1)
  action_with_noise = action + noise
  return np.clip(action_with_noise, action_low, action_high)

def discret_act_func(action):
  action_discret_group = np.zeros(action.shape[0])
  for i in range(action.shape[0]):
     action_per_unit = action[i]
     if (action_per_unit <= -0.5 and action_per_unit >= -1):
       action_discret = 0
     elif (action_per_unit > -0.5 and action_per_unit <= 0.5):
       action_discret = 1
     else:
       action_discret = 2
     action_discret_group[i] = action_discret
  return action_discret_group

def add_noise(action, action_low, action_high, action_noise):
  action_with_noise = np.zeros((action.shape[0], 1))
  for i in range(action.shape[0]):
    noise = action_noise()
    action_with_noise[i] = add_noise_per_unit(action[i], action_low, action_high, noise)
  return action_with_noise

def screenConcat(screen, num_agents):
  screen = screenReshape(screen)
  screen_final = screen
  if num_agents > 1:
    for i in range(num_agents-1):
      screen_final = np.concatenate((screen_final,screen),axis=0)
  return  screen_final

def screenReshape(screen):
  screen = np.array(screen)
  if (screen.shape[0] != 1):
    screen = screen.reshape((1, screen.shape[0] * screen.shape[1]))
  return screen

