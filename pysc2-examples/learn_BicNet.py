import numpy as np
np.set_printoptions(threshold = 1e6)

import os
import tempfile
import datetime
from absl import flags
import baselines.common.tf_util as U
from baselines import logger
import random

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

from common import common_group


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_HIT_POINT = features.SCREEN_FEATURES.unit_hit_points.index
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

_SELECT_POINT_ACT = [0]
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

FLAGS = flags.FLAGS

def learn(env,
          sess,
          actor,
          critic,
          replay_buffer,
          action_noise,
          num_exploring,
          max_timesteps=100000,
          train_freq=1,  #1
          batch_size=32,  #32
          print_freq=1,
          save_freq=10000, #10000
          gamma=1.0,
          target_network_update_freq=1,            #500,
          num_agents=9,
          num_baneling=4,
          num_zergling=6,
          unit_flag_friend=0.4, #48
          unit_flag_baneling=0.7,  #9
          unit_flag_zergling=1,  #105
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

  obs = env.reset()
  action_noise.reset()
  # obs,_ = common_group.init(env, obs)
  episode_rewards = [0.0]
  model_file_load = os.path.join(str(40000) + "_" + "model_segment_training/", "defeat_zerglings")
  # U.initialize()
  U.load_state(model_file_load, sess)

  unit_health = obs[0].observation["feature_units"]
  zero_fill = np.zeros((1,3))
  state_list = []
  group_list = []
  list_all_friend = []

  list_all_baneling = []
  list_all_zergling = []
  #extract location information
  for row in range(unit_health.shape[0]):
    list_per_friend = []
    list_per_baneling = []
    list_per_zergling = []

    coord_x = unit_health[row, 12]
    coord_y = unit_health[row, 13]

    if unit_health[row][0] == 48:
      list_per_friend.append(coord_x)
      list_per_friend.append(coord_y)
      list_per_friend.append(unit_flag_friend)
      list_all_friend.append(list_per_friend)

    if unit_health[row][0] == 9:
      list_per_baneling.append(coord_x/64)
      list_per_baneling.append(coord_y/64)
      list_per_baneling.append(unit_flag_baneling)
      list_all_baneling.append(list_per_baneling)

    if unit_health[row][0] == 105:
      list_per_zergling.append(coord_x/64)
      list_per_zergling.append(coord_y/64)
      list_per_zergling.append(unit_flag_zergling)
      list_all_zergling.append(list_per_zergling)

  coord_friend = np.array(list_all_friend)
  coord_baneling = np.array(list_all_baneling)
  coord_zergling = np.array(list_all_zergling)

  #zero padding
  coord_friend_non_norm = coord_friend
  while(coord_friend_non_norm.shape[0] < num_agents):
    coord_friend_non_norm = np.insert(coord_friend_non_norm, random.randint(0,coord_friend_non_norm.shape[0]), values=zero_fill, axis=0)

  coord_friend_pad = coord_friend_non_norm.copy()
  for i in range(coord_friend_non_norm.shape[0]):
    for j in range(coord_friend_non_norm.shape[1] - 1):
      coord_friend_pad[i][j] = coord_friend_non_norm[i][j]/64


  for i in range(coord_friend_pad.shape[0]):
    if(coord_friend_pad[i][2]!=0):
      group_list.append(i)

  while(coord_baneling.shape[0] < num_baneling):
    coord_baneling = np.insert(coord_baneling, random.randint(0, coord_baneling.shape[0]), values=zero_fill, axis=0)

  while(coord_zergling.shape[0] < num_zergling):
    coord_zergling = np.insert(coord_zergling, random.randint(0, coord_zergling.shape[0]), values=zero_fill, axis=0)

  # shared observation:location of enemy
  shared_obs = np.concatenate((coord_zergling, coord_baneling), axis=0)

  for i in range(num_agents):
    state_per_unit = np.concatenate((coord_friend_pad[i].reshape(1, coord_friend_pad[i].size), shared_obs), axis=0)
    state_list.append(state_per_unit)
  screen_expand = state_transform(state_list)


  with tempfile.TemporaryDirectory() as td:

    for t in range(max_timesteps):
      startTime = datetime.datetime.now()
      flag_end = False
      screen_input = np.expand_dims(screen_expand, axis=0)
      action = actor.predict(screen_input)[0]
      act_with_noise = np.clip(action + action_noise.get_noise(t-num_exploring), action_low, action_high)
      act = act_with_noise
      if (((_ATTACK_SCREEN in obs[0].observation["available_actions"]) and (_MOVE_SCREEN in obs[0].observation["available_actions"]))==False):
        id = random.choice(group_list)
        obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [_SELECT_POINT_ACT, (coord_friend_non_norm[id][0], coord_friend_non_norm[id][1])])])
        transition_add = False
      else:
        transition_add = True
        for i in (group_list):
          player = [coord_friend_non_norm[i][0], coord_friend_non_norm[i][1]]
          if(i < act.shape[0]):
            new_action = common_group.marine_action_continuous(player, act[i])
          else:
            new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
          obs = env.step_rewrite(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [_SELECT_POINT_ACT, (player[0], player[1])])])
          # if (_ATTACK_SCREEN in obs[0].observation["available_actions"]) or (_MOVE_SCREEN in obs[0].observation["available_actions"]):
          obs = env.step_rewrite(actions=new_action)
          # else:
          #   new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
          #   obs = env.step_rewrite(actions=new_action)
        # new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
        obs = env._step()
      actionEndTime = datetime.datetime.now()
      action_time_used = str(actionEndTime - startTime)
      print("t = %d, execute action time used = %s" % (t, action_time_used))
      done_final = obs[0].step_type == environment.StepType.LAST
      unit_health = obs[0].observation["feature_units"]
      zero_fill = np.zeros((1, 3))
      new_state_list = []
      group_list = []
      list_all_friend = []
      list_all_baneling = []
      list_all_zergling = []
      # extract location information
      checkListStart = datetime.datetime.now()
      for row in range(unit_health.shape[0]):
        list_per_friend = []
        list_per_baneling = []
        list_per_zergling = []

        coord_x = unit_health[row, 12]
        coord_y = unit_health[row, 13]

        if unit_health[row][0] == 48:
          list_per_friend.append(coord_x)
          list_per_friend.append(coord_y)
          list_per_friend.append(unit_flag_friend)
          list_all_friend.append(list_per_friend)

        if unit_health[row][0] == 9:
          list_per_baneling.append(coord_x / 64)
          list_per_baneling.append(coord_y / 64)
          list_per_baneling.append(unit_flag_baneling)
          list_all_baneling.append(list_per_baneling)

        if unit_health[row][0] == 105:
          list_per_zergling.append(coord_x / 64)
          list_per_zergling.append(coord_y / 64)
          list_per_zergling.append(unit_flag_zergling)
          list_all_zergling.append(list_per_zergling)

      checkListEnd = datetime.datetime.now()
      checkList_time_used = str(checkListEnd - checkListStart)
      print("t = %d, check list time used = %s" % (t, checkList_time_used))

      coord_friend = np.array(list_all_friend)
      coord_baneling = np.array(list_all_baneling)
      coord_zergling = np.array(list_all_zergling)

      # zero padding
      coord_friend_non_norm = coord_friend

      if (coord_friend_non_norm.shape[0] == 0):
        coord_friend_non_norm = np.zeros((num_agents, 3))

      amiZFStart = datetime.datetime.now()
      while (coord_friend_non_norm.shape[0] < num_agents):
        coord_friend_non_norm = np.insert(coord_friend_non_norm, random.randint(0, coord_friend_non_norm.shape[0]),
                                          values=zero_fill, axis=0)
      amiZFEnd = datetime.datetime.now()
      amiZF_time_used = str(amiZFEnd - amiZFStart)
      print("t = %d, friend zero fill time used = %s" % (t, amiZF_time_used))

      coord_friend_pad = coord_friend_non_norm.copy()
      for i in range(coord_friend_non_norm.shape[0]):
        for j in range(coord_friend_non_norm.shape[1] - 1):
          coord_friend_pad[i][j] = coord_friend_non_norm[i][j] / 64

      normEnd = datetime.datetime.now()
      norm_time_used = str(normEnd - amiZFEnd)
      print("t = %d, normalize time used = %s" % (t, norm_time_used))

      for i in range(coord_friend_pad.shape[0]):
        if (coord_friend_pad[i][2] != 0):
          group_list.append(i)

      checkGroupEnd = datetime.datetime.now()
      checkGruop_time_used = str(checkGroupEnd - normEnd)
      print("t = %d, check group time used = %s" % (t, checkGruop_time_used))

      if (coord_baneling.shape[0] == 0):
        coord_baneling = np.zeros((num_baneling, 3))
      if (coord_zergling.shape[0] == 0):
        coord_zergling = np.zeros((num_zergling, 3))

      zeroFillStart = datetime.datetime.now()
      while (coord_baneling.shape[0] < num_baneling):
        coord_baneling = np.insert(coord_baneling, random.randint(0, coord_baneling.shape[0]), values=zero_fill,
                                   axis=0)

      while (coord_zergling.shape[0] < num_zergling):
        coord_zergling = np.insert(coord_zergling, random.randint(0, coord_zergling.shape[0]), values=zero_fill,
                                   axis=0)
      zeroFillEnd = datetime.datetime.now()
      zeroFill_time_used = str(zeroFillEnd - zeroFillStart)
      print("t = %d, zerg zero fill time used = %s" % (t, zeroFill_time_used))


      # shared observation:location of enemy
      shared_obs = np.concatenate((coord_zergling, coord_baneling), axis=0)

      for i in range(num_agents):
        state_per_unit = np.concatenate((coord_friend_pad[i].reshape(1, coord_friend_pad[i].size), shared_obs),
                                        axis=0)
        new_state_list.append(state_per_unit)
      new_screen_expand = state_transform(new_state_list)

      stateCalcEnd = datetime.datetime.now()
      stateCalc_time_used = str(stateCalcEnd - zeroFillEnd)
      print("t = %d, state calc time used = %s" % (t, stateCalc_time_used))

      if (done_final):
        flag_end = True

      player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
      friend_y, friend_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if (len(friend_x) == 0):
        flag_end = True
      rew = obs[0].reward

      if(transition_add):
        rew_expand = np.zeros((num_agents, 1))
        for i in (group_list):
          if i < num_agents:
            rew_expand[i] = rew  # rew_per_unit
        replay_buffer.add(screen_expand, act_with_noise, rew_expand, flag_end,
                          new_screen_expand)  # [group0:[num_traces, trace.dimension], group1, ... group8]

      episode_rewards[-1] += rew  # rew.sum(axis=0)


      screen_expand = new_screen_expand

      updateStateEndTime = datetime.datetime.now()
      updateState_time_used = str(updateStateEndTime - actionEndTime)
      print("t = %d, update state time used = %s" % (t, updateState_time_used))

      if (flag_end):
        initStartTime = datetime.datetime.now()
        reward = episode_rewards[-1]
        print("Episode Reward : %s" % reward)
        obs = env.reset()
        action_noise.reset()
        # obs, _ = common_group.init(env, obs)
        print('num_episodes is', len(episode_rewards))
        episode_rewards.append(0.0)

        unit_health = obs[0].observation["feature_units"]
        zero_fill = np.zeros((1, 3))
        state_list = []
        group_list = []
        list_all_friend = []
        list_all_baneling = []
        list_all_zergling = []
        # extract location information
        for row in range(unit_health.shape[0]):
          list_per_friend = []
          list_per_baneling = []
          list_per_zergling = []

          coord_x = unit_health[row, 12]
          coord_y = unit_health[row, 13]

          if unit_health[row][0] == 48:
            list_per_friend.append(coord_x)
            list_per_friend.append(coord_y)
            list_per_friend.append(unit_flag_friend)
            list_all_friend.append(list_per_friend)

          if unit_health[row][0] == 9:
            list_per_baneling.append(coord_x / 64)
            list_per_baneling.append(coord_y / 64)
            list_per_baneling.append(unit_flag_baneling)
            list_all_baneling.append(list_per_baneling)

          if unit_health[row][0] == 105:
            list_per_zergling.append(coord_x / 64)
            list_per_zergling.append(coord_y / 64)
            list_per_zergling.append(unit_flag_zergling)
            list_all_zergling.append(list_per_zergling)

        coord_friend = np.array(list_all_friend)
        coord_baneling = np.array(list_all_baneling)
        coord_zergling = np.array(list_all_zergling)

        # zero padding
        coord_friend_non_norm = coord_friend
        while (coord_friend_non_norm.shape[0] < num_agents):
          coord_friend_non_norm = np.insert(coord_friend_non_norm, random.randint(0, coord_friend_non_norm.shape[0]),
                                            values=zero_fill, axis=0)

        coord_friend_pad = coord_friend_non_norm.copy()
        for i in range(coord_friend_non_norm.shape[0]):
          for j in range(coord_friend_non_norm.shape[1] - 1):
            coord_friend_pad[i][j] = coord_friend_non_norm[i][j] / 64

        for i in range(coord_friend_pad.shape[0]):
          if (coord_friend_pad[i][2] != 0):
            group_list.append(i)

        while (coord_baneling.shape[0] < num_baneling):
          coord_baneling = np.insert(coord_baneling, random.randint(0, coord_baneling.shape[0]), values=zero_fill,
                                     axis=0)

        while (coord_zergling.shape[0] < num_zergling):
          coord_zergling = np.insert(coord_zergling, random.randint(0, coord_zergling.shape[0]), values=zero_fill,
                                     axis=0)

        # shared observation:location of enemy
        shared_obs = np.concatenate((coord_zergling, coord_baneling), axis=0)

        for i in range(num_agents):
          state_per_unit = np.concatenate((coord_friend_pad[i].reshape(1, coord_friend_pad[i].size), shared_obs),
                                          axis=0)
          state_list.append(state_per_unit)
        screen_expand = state_transform(state_list)

        initEndTime = datetime.datetime.now()
        init_time_used = str(initEndTime - initStartTime)
        print("t = %d, init time used = %s" % (t, init_time_used))

      if (t > num_exploring)and (t % train_freq == 0):      #t % train_freq == 0:
        trainStartTime = datetime.datetime.now()
        print("training starts")
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(batch_size)     #[group0:[batch_size, trace.dimension], group1, ... group8]
        target_q = r_batch + gamma * critic.predict_target(s2_batch, actor.predict_target(s2_batch))
        predicted_q_value, _ = critic.train(s_batch, a_batch,
                                            np.reshape(target_q, (batch_size, num_agents, 1)))
        a_outs = actor.predict(s_batch)  # a_outs和a_batch是完全相同的
        grads = critic.action_gradients(s_batch, a_outs)  # delta Q对a的导数
        actor.train(s_batch, grads)  # 这里会计算a对θ的导数和最后的梯度
        trainEndTime = datetime.datetime.now()
        train_time_used = str(trainEndTime - trainStartTime)
        print("t = %d, training time used = %s" % (t, train_time_used))

      if (t > num_exploring) and (t % target_network_update_freq == 0):#t % target_network_update_freq == 0:
        # Update target network periodically.
        updNetStartTime = datetime.datetime.now()
        actor.update_target_network()
        critic.update_target_network()
        updNetEndTime = datetime.datetime.now()
        updNet_time_used = str(updNetEndTime - updNetStartTime)
        print("t = %d, update net time used = %s" % (t, updNet_time_used))

      if (t > num_exploring) and ((t - num_exploring) % save_freq == 0):
        saveStartTime = datetime.datetime.now()
        model_file_save = os.path.join(str(t+40000) + "_" + "model_segment_training/", "defeat_zerglings")
        U.save_state(model_file_save)
        replay_buffer.save()
        saveEndTime = datetime.datetime.now()
        save_time_used = str(saveEndTime - saveStartTime)
        print("t = %d, save model time used = %s" % (t, save_time_used))

      elif (t == max_timesteps - 1):
        model_file_save = os.path.join(str(t+40000) + "_" + "model_segment_training/", "defeat_zerglings")
        U.save_state(model_file_save)

      # mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
      num_episodes = len(episode_rewards)
      if flag_end and print_freq is not None and len(
          episode_rewards) % print_freq == 0:
        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", num_episodes)
        logger.record_tabular("reward", reward)
        # logger.record_tabular("mean 100 episode reward",
        #                       mean_100ep_reward)
        logger.dump_tabular()

      endTime = datetime.datetime.now()
      time_used = str(endTime - startTime)
      print("t = %d, time used = %s" % (t,time_used))



# def add_noise_per_unit(action, action_low, action_high, noise):
#   # s = np.random.normal(0, 0.1)
#   action_with_noise = action + noise
#   return np.clip(action_with_noise, action_low, action_high)

# def discret_act_func(action):
#   action_discret_group = np.zeros(action.shape[0])
#   for i in range(action.shape[0]):
#      action_per_unit = action[i]
#      if (action_per_unit <= -0.5 and action_per_unit >= -1):
#        action_discret = 0
#      elif (action_per_unit > -0.5 and action_per_unit <= 0.5):
#        action_discret = 1
#      else:
#        action_discret = 2
#      action_discret_group[i] = action_discret
#   return action_discret_group

# def add_noise(action, action_low, action_high, action_noise, t):
#
#   action_with_noise = np.zeros((action.shape[0], action.shape[1]))
#   # noise = action_noise()
#   noise = action_noise.get_noise(t)
#   # noise = np.random.rand() * 2 - 1
#   for i in range(action.shape[0]):
#       for j in range(action.shape[1]):
#          action_with_noise[i][j] = add_noise_per_unit(action[i][j], action_low, action_high, noise)
#          # print("nine agent original action[" + str(i) + "][" + str(j) + "] is :", action[i][j])
#          # print("nine agent noise[" + str(i) + "][" + str(j) + "] is :", noise)
#          # print("nine agent action add noise[" + str(i) + "][" + str(j) + "] is :", action_with_noise[i][j])
#   return action_with_noise

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

def state_transform(state_list):
  screen_final = np.array([])
  for i in range(len(state_list)):
    screen = state_list[i]
    screen = screen.reshape((1, screen.shape[0] * screen.shape[1]))
    if(screen_final.size!=0):
      screen_final = np.concatenate((screen_final, screen), axis=0)
    else:
      screen_final = screen
  return  screen_final