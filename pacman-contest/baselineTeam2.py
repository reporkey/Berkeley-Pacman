# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np
from game import Grid



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class ValueiterationAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, gameState,discount = 0.9):                                               #default parameters
        self.gameState= gameState
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()                              #walls是一个真值列表 wallsposition才是墙的坐标
        self.foods = self.getFood(gameState).asList()                  #food和walls的获取需要用不同的方法。可能跟他其他文件的相关定义有关
        self.capsules = self.getCapsules(gameState)
        width, height = self.walls.width, self.walls.height
        #print(width,height)
        wallsposition=[]                                                                   #the location of walls
        for x in range(width):
            for y in range(height):
                if self.walls[x][y]==True:
                    wallsposition.append([x,y])
        print(wallsposition)
        #print(self.states)
        self.rewards = np.zeros((height, width), dtype=None)                            #set every initial-reward 0. It also means the cost of move is 0
        for i in self.foods:                                                  #set reward of each food as 10
            for x in range(width):
                for y in range(height):
                    if (x,y)==i:
                      self.rewards[height-1-y][x]=10                            #set reward of each food as 10
        for i in self.capsules:                                              #set reward of each capsule as 100
            for x in range(width):
                for y in range(height):
                    if (x,y)==i:
                      self.rewards[height-1-y][x]=100
        for i in wallsposition:  # set reward of each capsule as 100
            for x in range(width):
                for y in range(height):
                    if [x, y] == i:
                        self.rewards[height-1 - y][x] = None
        print(self.rewards)
        #print(self.values)
        CaptureAgent.registerInitialState(self, gameState)
        #print(self.start)
        #nextState =  gameState.generateSuccessor(gameState, gameState.getLegalActions())
        self.poslist = [index for index, x in np.ndenumerate(self.rewards) if x >=0]
        print(self.poslist)

        self.discount = discount
        CaptureAgent.registerInitialState(self, gameState)
        self.values = np.zeros_like(self.rewards, dtype=np.float32)

        valueiterationAgent = ValueiterationAgent(self.index)
        for _ in range(10):
          valueiterationAgent.update_values()

    def getSuccessor(self):
        self.getSuccessor = dict()
        for i, j in self.poslist:
            next_states = list()
            if (i - 1, j) in self.poslist:
                next_states.append((i - 1, j))
            if (i + 1, j) in self.poslist:
                next_states.append((i + 1, j))
            if (i, j - 1) in self.poslist:
                next_states.append((i, j - 1))
            if (i, j + 1) in self.poslist:
                next_states.append((i, j + 1))
            self.getSuccessor[(i, j)] = next_states

    def get_reward(self, i, j):
        return self.rewards[i, j]

    def best_value_func(self, i, j):
      return self.get_reward(i, j) + self.discount * max(self.next_states_expected_value(i, j))

    def update_values(self):
      for i, j in self.poslist:
        self.values[i, j] = self.best_value_func(i, j)

    def next_states_expected_value(self, i, j):
      next_values = []
      ns_num = len(self.getSuccessor[(i, j)])
      if ns_num == 1:
        ps = [1]
      elif ns_num == 2:
        ps = [0.5, 0.5]
      elif ns_num == 3:
        ps = [1 / 3, 1 / 3, 1 / 3]
      elif ns_num == 4:
        ps = [0.25, 0.25, 0.25, 0.25]
      for next_index in self.getSuccessor[(i, j)]:
        other_index = [index for index in self.getSuccessor[(i, j)] if index != next_index]
        ns_index = [next_index] + other_index
        values = [self.values[i, j] for i, j in ns_index]
        next_values.append(np.multiply(values, ps).sum())
      return next_values

    def best_policy(self, i, j):
      next_states = self.getSuccessor[(i, j)]
      best_state_index = np.argmax(self.next_states_expected_value(i, j))
      best_state = next_states[best_state_index]
      return self.best_policy(best_state[0], best_state[1])
      print("(%d, %d)" % (i, j))
      return


"""
gameState=CaptureAgent.registerInitialState()
env=Env(gameState)
valueiterationAgent=ValueiterationAgent(gameState,env,discount = 0.9)
for _ in range(10):
    valueiterationAgent.update_values()
"""
class OffensiveReflexAgent(ValueiterationAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ValueiterationAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


