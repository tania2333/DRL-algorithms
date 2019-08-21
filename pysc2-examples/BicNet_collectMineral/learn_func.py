import numpy as np
np.set_printoptions(threshold = 1e6)

import math
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
          output_len=4,
          # num_baneling=4,
          # num_zergling=6,
          # unit_flag_friend=0.4, #48
          # unit_flag_baneling=0.7,  #9
          # unit_flag_zergling=1,  #105
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
  episode_rewards = [0.0]
  obs,_ = common_group.init(env, obs)
  # model_file_load = os.path.join(str(40000) + "_" + "model_segment_training/", "defeat_zerglings")
  # U.load_state(model_file_load, sess)
  U.initialize()
  min = 5
  punish = -0.01
  eps_time = 1

  #求出screen_expand
  player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
  screen = np.zeros((player_relative.shape[0]-9, player_relative.shape[1]))
  for i in range(player_relative.shape[0]-9):
    for j in range(player_relative.shape[1]):
      screen[i,j] = round(player_relative[i,j]/3, 1)
  screen_expand = screenConcat(screen, num_agents)

  #选择，以便MOVE_SCREEN是available的
  obs = env.step(
    actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

  player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  # 对player_x和player_y的处理,保证其含有两个元素,对应两个agent的位置
  # 智能体不存在的情况，一般不会出现，因为地图不是敌对双方的设置
  if (len(player_x) == 0):
    player_x = np.array([0])
    player_y = np.array([0])

  #两个智能体重合于一点的情况
  if (len(player_x) == 1):
    player_x = np.append(player_x, player_x[0])
    player_y = np.append(player_y, player_y[0])


  pos_agent1_target = [player_x[0], player_y[0]]
  pos_agent2_target = [player_x[1], player_y[1]]

  with tempfile.TemporaryDirectory() as td:

    for t in range(max_timesteps):
      startTime = datetime.datetime.now()
      #输入观察，得到动作
      screen_input = np.expand_dims(screen_expand, axis=0)

      action = actor.predict(screen_input)[0] # (2, 4)
      rnn_out = actor.rnn_out_pre(screen_input)
      # action[0] = MaxMinNormalization(action[0], getMax(action[0]), getMin(action[0]))
      # action[1] = MaxMinNormalization(action[1], getMax(action[1]), getMin(action[1]))
      act_with_noise = np.clip(action + action_noise.get_noise(t-num_exploring), action_low, action_high)
      act_prob = (act_with_noise + 1) / 2  #act_with_noise
      # act_prob_sum = act_prob.sum(axis=1)
      act_index = [0,1,2,3]
      # if(act_prob_sum[0] == 0):
      #   prob = (np.array(act_prob[0]) + 1) / len(act_prob[0])
      # else:
      #   prob = act_prob[0]/act_prob_sum[0]
      #
      # a1 = np.random.choice(np.array(act_index), p=prob.ravel())
      #
      # if (act_prob_sum[1] == 0):
      #   prob = (np.array(act_prob[1]) + 1) / len(act_prob[1])
      # else:
      #   prob = act_prob[1] / act_prob_sum[1]
      #
      # a2 = np.random.choice(np.array(act_index), p=prob.ravel())
      # a1 = act_with_noise[0]
      # a2 = act_with_noise[1]
      arr_m = act_prob[0]
      max_m = -1

      for i in range(len(arr_m)):
        if(arr_m[i]>max_m):
          max_m = arr_m[i]
          idx_max_m = i

      arr_n = act_prob[1]
      max_n = -1

      for j in range(len(arr_n)):
        if (arr_n[j] > max_n):
          max_n = arr_n[j]
          idx_max_n = j

      #选择概率最大的动作
      a1 = idx_max_m
      a2 = idx_max_n

      #动作执行
      pos_agent1 = [player_x[0], player_y[0]]
      pos_agent2 = [player_x[1], player_y[1]]

      diff_1toTarget1 = (pos_agent1_target[0] - pos_agent1[0])*(pos_agent1_target[0] - pos_agent1[0])+(pos_agent1_target[1] - pos_agent1[1])*(pos_agent1_target[1] - pos_agent1[1])
      diff_2toTarget1 = (pos_agent1_target[0] - pos_agent2[0]) * (pos_agent1_target[0] - pos_agent2[0]) + (pos_agent1_target[1] - pos_agent2[1]) * (pos_agent1_target[1] - pos_agent2[1])

      diff_1toTarget2 = (pos_agent2_target[0] - pos_agent1[0]) * (pos_agent2_target[0] - pos_agent1[0]) + (pos_agent2_target[1] - pos_agent1[1]) * (pos_agent2_target[1] - pos_agent1[1])
      diff_2toTarget2 = (pos_agent2_target[0] - pos_agent2[0]) * (pos_agent2_target[0] - pos_agent2[0]) + (pos_agent2_target[1] - pos_agent2[1]) * (pos_agent2_target[1] - pos_agent2[1])

      if((diff_1toTarget1 > diff_2toTarget1) and (diff_1toTarget2 < diff_2toTarget2)):
        pos_agent1 = [player_x[1], player_y[1]]
        pos_agent2 = [player_x[0], player_y[0]]

      # 如果本来就位于边缘，还往边缘方向跑，就给惩罚
      pos_agent1_target,punish_1 = obtainTargetPos(a1, pos_agent1)
      pos_agent2_target, punish_2 = obtainTargetPos(a2, pos_agent2)

      player_relative_old = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
      mineral_y_old, mineral_x_old = (player_relative_old == _PLAYER_NEUTRAL).nonzero()
      if (len(mineral_x_old) == 0):
        mineral_x_old = np.array([0])
        mineral_y_old = np.array([0])

      obs = env.step_rewrite(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [_SELECT_POINT_ACT, pos_agent1])])
      obs = env.step_rewrite(actions=[sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, pos_agent1_target])])
      obs = env.step_rewrite(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [_SELECT_POINT_ACT, pos_agent2])])
      obs = env.step_rewrite(actions=[sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, pos_agent2_target])])
      obs = env._step()

      flag_end = obs[0].step_type == environment.StepType.LAST
      rew = obs[0].reward
      #得到新的观察
      player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
      new_screen = np.zeros((player_relative.shape[0]-9, player_relative.shape[1]))
      for i in range(player_relative.shape[0]-9):
        for j in range(player_relative.shape[1]):
          new_screen[i, j] = round(player_relative[i, j] / 3, 1)
      new_screen_expand = screenConcat(new_screen, num_agents)


      #得到新的矿到智能体距离总和
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

      # 对player_x和player_y的处理,保证其含有两个元素,对应两个agent的位置
      # 智能体不存在的情况，一般不会出现，因为地图不是敌对双方的设置
      if (len(player_x) == 0):
        player_x = np.array([0])
        player_y = np.array([0])

      # 两个智能体重合于一点的情况
      if (len(player_x) == 1):
        player_x = np.append(player_x, player_x[0])
        player_y = np.append(player_y, player_y[0])

      # 求智能体到step之前最近的矿的距离,如果小于最小值，认为已经采到矿
      reward_dist_a1 = False
      reward_dist_a2 = False
      for i in range(len(mineral_x_old)):
        dist_agent1 = (mineral_x_old[i] - player_x[0]) * (mineral_x_old[i] - player_x[0]) + (mineral_y_old[i] - player_y[0]) * (
                  mineral_y_old[i] - player_y[0])
        dist_agent2 = (mineral_x_old[i] - player_x[1]) * (mineral_x_old[i] - player_x[1]) + (mineral_y_old[i] - player_y[1]) * (
                  mineral_y_old[i] - player_y[1])
        if (dist_agent1 < min and rew > 0):
          reward_dist_a1 = True
          break
        if (dist_agent2 < min and rew > 0):
          reward_dist_a2 = True
          break

      # 根据前后矿到智能体的距离计算各自奖励
      rew_expand = np.zeros((num_agents, 1))

      #collect mineral reward
      if(reward_dist_a1 and rew>0):
        rew_expand[0] = rew
      if(reward_dist_a2 and rew>0):
        rew_expand[1] = rew

      # if(reward_dist_a1 or reward_dist_a2 or rew==1):
      #   rew_expand[0] += rew*10
      #   rew_expand[1] += rew*10

      # 每一步给一个惩罚值
      if(punish_1):
        rew_expand[0] += -10   #碰壁给一个惩罚
      rew_expand[0] += punish * eps_time

      if (punish_2):
        rew_expand[1] += -10
      rew_expand[1] += punish * eps_time

      # if (punish_1 or punish_2):
      #   rew_expand[0] += -10
      #   rew_expand[1] += -10


      replay_buffer.add(screen_expand, act_with_noise, rew_expand, flag_end,
                      new_screen_expand)

      episode_rewards[-1] += rew  # rew.sum(axis=0)

      # 将新的观察作为当前观察
      screen_expand = new_screen_expand
      eps_time += 1

      if (flag_end):
        eps_time = 1
        reward = episode_rewards[-1]
        print("Episode Reward : %s" % reward)
        obs = env.reset()
        action_noise.reset()
        print('num_episodes is', len(episode_rewards))
        episode_rewards.append(0.0)

        #得到初始观察
        player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
        screen = np.zeros((player_relative.shape[0]-9, player_relative.shape[1]))
        for i in range(player_relative.shape[0]-9):
          for j in range(player_relative.shape[1]):
            screen[i, j] = round(player_relative[i, j] / 3, 1)
        screen_expand = screenConcat(screen, num_agents)


        #选中全部
        obs = env.step(
          actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

        # 求出最开始矿到每个智能体的距离和
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

        # 对player_x和player_y的处理,保证其含有两个元素,对应两个agent的位置
        if (len(player_x) == 0):
          player_x = np.array([0])
          player_y = np.array([0])

        if (len(player_x) == 1):
          player_x = np.append(player_x, player_x[0])
          player_y = np.append(player_y, player_y[0])

        pos_agent1_target = [player_x[0], player_y[0]]
        pos_agent2_target = [player_x[1], player_y[1]]

      if (t > num_exploring)and (t % train_freq == 0):      #t % train_freq == 0:
        # trainStartTime = datetime.datetime.now()
        print("training starts")
        s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(batch_size)     #[group0:[batch_size, trace.dimension], group1, ... group8]
        target_q = r_batch + gamma * critic.predict_target(s2_batch, actor.predict_target(s2_batch))
        rnn_c_out = critic.predict_target_rnn(s2_batch, actor.predict_target(s2_batch))
        predicted_q_value, _ = critic.train(s_batch, a_batch,
                                            np.reshape(target_q, (batch_size, num_agents,output_len)))
        a_outs = actor.predict(s_batch)  # a_outs和a_batch是完全相同的
        grads = critic.action_gradients(s_batch, a_outs)  # delta Q对a的导数
        actor.train(s_batch, grads)  # 这里会计算a对θ的导数和最后的梯度


      if (t > num_exploring) and (t % target_network_update_freq == 0):#t % target_network_update_freq == 0:
        actor.update_target_network()
        critic.update_target_network()


      if (t > num_exploring) and ((t - num_exploring) % save_freq == 0):
        # saveStartTime = datetime.datetime.now()
        model_file_save = os.path.join(str(t) + "_" + "model_segment_training2/", "defeat_zerglings")
        U.save_state(model_file_save)
        replay_buffer.save()


      elif (t == max_timesteps - 1):
        model_file_save = os.path.join(str(t) + "_" + "model_segment_training2/", "defeat_zerglings")
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

def obtainTargetPos(act, pos_old):
    # 动作网络输出为4方向
    player_x = pos_old[0]
    player_y = pos_old[1]
    pos = pos_old.copy()
    punish = False
    if(act == 0):
        if(player_y > 1):    #up
            pos = [player_x, player_y - 2]
        else:
          punish = True
    elif(act == 1):  #left
        if (player_x > 1):
            pos = [player_x - 2, player_y]
        else:
          punish = True
    elif(act == 2):  #down
        if (player_y < 22): #22  31
            pos = [player_x, player_y + 2]
        else:
          punish = True
    else:
        if (player_x < 30):  #22  31
            pos = [player_x + 2, player_y]
        else:
          punish = True
    # degree = (act + 1) * math.pi
    # delta_x = math.cos(degree)
    # delta_y = math.sin(degree)
    # distance = 2
    # pos = pos_old.copy()
    # pos[0] += round(distance * delta_x)
    # pos[1] -= round(distance * delta_y)
    # punish = False
    # if(pos[0]>30):
    #   pos[0] = 30
    #   punish = True
    # elif(pos[0]<1):
    #   pos[0] = 1
    #   punish = True
    #
    # if(pos[1]>22):
    #   pos[1] = 22
    #   punish = True
    # elif(pos[1]<1):
    #   pos[1] = 1
    #   punish = True

    return pos, punish

def getMax(x):
  max = x[0]
  for i in range(len(x)):
    if(x[i]>max):
      max=x[i]
  return max

def getMin(x):
  min=x[0]
  for i in range(len(x)):
    if(x[i]<min):
      min = x[i]
  return min

def MaxMinNormalization(x, Max, Min):
  if(Max != Min):
    x = (x - Min) / (Max - Min)
  return x