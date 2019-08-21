import numpy as np
np.set_printoptions(threshold = 1e6)
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env import environment
import math
import random
from mineral.tsp2 import multistart_localsearch, mk_matrix, distL2

_DENSITY_UNIT = features.SCREEN_FEATURES.unit_density.index
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

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id  #control_group(action, action_space, control_group_act, control_group_id):
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# _STOP_QUICK = actions.FUNCTIONS.stop_quick.id      #(action, action_space, ability_id, queued)
_NOT_QUEUED = 0
_SELECT_ALL = 0


def init(env, obs):
    unitlist = unit_position(obs, 1)
    for i in range(unitlist.__len__() if unitlist.__len__() < 11 else 10):
        # xy_per_marine[str(i)] = unitlist[i]
        obs = env.step_rewrite(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[0], [unitlist[i][1], unitlist[i][0]]])
        ])
        obs = env.step_rewrite(actions=[
            sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP,
                                     [[_CONTROL_GROUP_SET], [i]])
        ])
    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
    obs = env.step(actions=new_action)
    done_init_group = obs[0].step_type == environment.StepType.LAST
    return obs,done_init_group       #,xy_per_marine

def unit_position(obs, flag):
    # by the screen's density map and pix number
    # to cal the army(enemy)'s position
    #input  1 army      0 enemy
    # output : list=[y, x]

  list_unit = []
  screen_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
  screen_density_unit = obs[0].observation["feature_screen"][_DENSITY_UNIT]  # How many units are in this pixel
  enemy_y, enemy_x = (screen_relative == _PLAYER_HOSTILE).nonzero()
  player_y, player_x = (screen_relative == _PLAYER_FRIENDLY ).nonzero()

  unit_y, unit_x = (player_y,player_x) if flag == 1 else (enemy_y,enemy_x)  #which unit to get
  unit_y = unit_y.tolist()#【8,8,9,9,....】
  unit_x = unit_x.tolist() #【[14,14,20,20,....】
  while(len(unit_y) > 0 ) and (len(unit_x) > 0):
    try:
        pos = [unit_y[0], unit_x[0]]           #用横纵坐标中的第一个来初始化
    except Exception:
        print("the unit is {} {}".format(unit_y,unit_x))
    _record = np.array([ [pos[0], pos[1]], [pos[0], pos[1]+1],     #与pos相连的4个坐标作为一个agent的记录
                        [pos[0]+1, pos[1]], [pos[0]+1, pos[1]+1] ])

    #make sure there are three pix is the same type
    cnt = 0
    for j in range(4):
      pos = [_record[j][0], _record[j][1]]
      if pos in _record:
        cnt += 1
    if cnt < 4:
      break

    for j in range(len(_record)):
      if(screen_density_unit[_record[j][0], _record[j][1]] > 1):
        screen_density_unit[_record[j][0], _record[j][1]] -= 1
        np.delete(_record, j, axis=0)
      else:
        if _record[j][0] in unit_y:
          unit_y.remove(_record[j][0])
        if _record[j][1] in unit_x:
          unit_x.remove(_record[j][1])
    if _record.size > 5:  #把重合的像素点删除之后，该agent还剩至少2个像素点，1个像素点=4 coord
      list_unit.append(_record[0])

  return list_unit

def update_group_list(obs):
    control_groups = obs[0].observation["control_groups"]
    group_count = 0
    group_list = []
    for id, group in enumerate(control_groups):
        if (group[0] != 0):
            group_count += 1
            group_list.append(id)
    return group_list


def check_group_list(env, obs):
    error = False
    control_groups = obs[0].observation["control_groups"]   # print('control_groups',control_groups)
    army_count = 0
    for id, group in enumerate(control_groups):
        if (group[0] == 48):       #[[48,1],[0,0], [48,1], ..]-> type (10,2) ->(leader_unit_type, count)
            army_count += group[1]              #不为0的组中包括的所有单位数加在一起
            if (group[1] != 1):
                # print("group error group_id : %s count : %s" % (id, group[1]))
                error = True
                return error
    if (army_count != env._obs[0].observation.player_common.army_count):
        error = True
        # print("army_count %s !=  %s env._obs.observation.player_common.army_count "
        #      % (army_count, env._obs.observation.player_common.army_count))

    return error


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


def shift(direction, number, matrix):
    ''' shift given 2D matrix in-place the given number of rows or columns
      in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
  '''
    if direction in (UP):
        matrix = np.roll(matrix, -number, axis=0)
        matrix[number:, :] = -2
        return matrix
    elif direction in (DOWN):
        matrix = np.roll(matrix, number, axis=0)
        matrix[:number, :] = -2
        return matrix
    elif direction in (LEFT):
        matrix = np.roll(matrix, -number, axis=1)
        matrix[:, number:] = -2
        return matrix
    elif direction in (RIGHT):
        matrix = np.roll(matrix, number, axis=1)
        matrix[:, :number] = -2
        return matrix
    else:
        return matrix


def select_marine(env, obs, group_id):
  # obs = env.step(actions = sc2_actions.FunctionCall(_STOP_QUICK, [[_NOT_QUEUED]]))
  player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
  screen = player_relative
  Remove = False
  # group_list = update_group_list(obs)

  # if (check_group_list(env, obs)):
  #   obs = init(env, obs)
  #   group_list = update_group_list(obs)

  # if(len(group_list) == 0):
  #   obs = init(env, player_relative, obs)
  #   group_list = update_group_list(obs)

  player = []
                                            #随机选择一个组
    # If there is no marine in danger, select random
  # while (len(group_list) > 0):
  obs = env.step_rewrite(actions=[
      sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[
          _CONTROL_GROUP_RECALL
      ], [int(group_id)]])
  ])
  selected = obs[0].observation["feature_screen"][_SELECTED]
  player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
  if (len(player_y) > 0):
    # player = [int(player_x.mean()), int(player_y.mean())]
    player = [player_x[0], player_y[0]]
    # break
  else:
    # print("group_id", group_id)
    # print("group_list",group_list)
    # group_list.remove(group_id)
    Remove = True
    # break
  done = obs[0].step_type == environment.StepType.LAST
  return obs, screen, player, Remove, done

def marine_action(env, obs, player, action):
    player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

    enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

    closest, min_dist = None, None

    distance = 0

    flag_punish = False

    if (len(player) == 2):
        for p in zip(enemy_x, enemy_y):
            dist = np.linalg.norm(np.array(player) - np.array(p))
            if not min_dist or dist < min_dist:
                closest, min_dist = p, dist

    player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
    friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    closest_friend, min_dist_friend = None, None
    if (len(player) == 2):
        for p in zip(friendly_x, friendly_y):
            dist = np.linalg.norm(np.array(player) - np.array(p))  # 求整体的矩阵元素平方和，再开根号
            if not min_dist_friend or dist < min_dist_friend:
                closest_friend, min_dist_friend = p, dist

    diff_playerToEnemy = np.array(player) - np.array(closest)
    diff_playerToEnemy = np.linalg.norm(diff_playerToEnemy)
    if(diff_playerToEnemy > 20):
        flag_punish = True

    if (closest == None):

        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

    elif (action == 0 and closest_friend != None and min_dist_friend < 3):
        # Friendly marine is too close => Sparse!
        diff = np.array(player) - np.array(closest_friend)

        norm = np.linalg.norm(diff)

        if (norm != 0):
            diff = diff / norm

        coord = np.array(player) + diff * 4
        # print('coord',coord)
        if (coord[0] < 0):
            coord[0] = 0
        elif (coord[0] > 63):
            coord[0] = 63

        if (coord[1] < 0):
            coord[1] = 0
        elif (coord[1] > 63):
            coord[1] = 63

        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    # elif (action <= 1):  # Attack
    elif (action == 1):  # Attack

        # nearest enemy

        coord = closest

        new_action = [
            sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
        ]

        # print("action : %s Attack Coord : %s" % (action, coord))

    elif (action == 2):  # Oppsite direcion from enemy - to get away from the enemy

        # nearest enemy opposite

        diff = np.array(player) - np.array(closest)

        norm = np.linalg.norm(diff)
        distance = norm
        if (norm != 0):
            diff = diff / norm

        coord = np.array(player) + diff * 7

        if (coord[0] < 0):
            coord[0] = 0
        elif (coord[0] > 63):
            coord[0] = 63

        if (coord[1] < 0):
            coord[1] = 0
        elif (coord[1] > 63):
            coord[1] = 63

        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    else:
        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]  # it means no enemy

        # print("action : %s Back Coord : %s" % (action, coord))

    return obs, new_action, distance, flag_punish
def marine_action_continuous(player, action):
    degree = (action[1] + 1) * math.pi
    distance = (action[2] + 1) * 31.5  # [0,63]
    delta_x = math.cos(degree)
    delta_y = math.sin(degree)
    coord = np.array(player)
    coord[0] += round(distance * delta_x)
    coord[1] -= round(distance * delta_y)

    if (coord[0] < 0):
        coord[0] = 0
    elif (coord[0] > 63):
        coord[0] = 63

    if (coord[1] < 0):
        coord[1] = 0
    elif (coord[1] > 63):
        coord[1] = 63

    prob_attack = (action[0] + 1)/2   # [0, 100]
    p = np.array([1-prob_attack, prob_attack])

    index = np.random.choice([0,1], p=p.ravel())
    if (index == 1):
    # if(random_index([prob_attack, 1-prob_attack]) == 0):
        new_action = [
            sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
        ]
    else:
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]


    return new_action

def random_index(rate):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index


