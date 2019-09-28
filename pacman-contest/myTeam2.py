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

class Env:
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, gameState):                                                     #default parameters
        self.gameState= gameState
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()                              #walls是一个真值列表 wallsposition才是墙的坐标
        self.foods = self.getFood(gameState).asList()                  #food和walls的获取需要用不同的方法。可能跟他其他文件的相关定义有关
        self.capsules = self.getCapsules(gameState)
        width, height = self.walls.width, self.walls.height
        self.rewards = np.zeros((height, width), dtype=None)                            #set every initial-reward 0. It also means the cost of move is 0
        #print(width,height)
        wallsposition=[]                                                                   #the location of walls
        for x in range(width):
            for y in range(height):
                if self.walls[x][y]==True:
                    wallsposition.append([x,y])
        print(wallsposition)
        #print(self.states)
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
        self.values = util.Counter()                                            # A Counter is a dict with default 0
        #print(self.values)
        CaptureAgent.registerInitialState(self, gameState)
        #print(self.start)
        #nextState =  gameState.generateSuccessor(gameState, gameState.getLegalActions())
        self.poslist = [index for index, x in np.ndenumerate(self.rewards) if x == 0]
        print(self.poslist)

    def getSuccessor(self, i, j):
        successors = []
        if (i - 1, j) in self.poslist:
            successors.append((i - 1, j))
        if (i + 1, j) in self.poslist:
            successors.append((i + 1, j))
        if (i, j - 1) in self.poslist:
            successors.append((i, j - 1))
        if (i, j + 1) in self.poslist:
            successors.append((i, j + 1))
        return successors

    def get_reward(self, i, j):
        return self.rewards[i, j]

    def get_states(self):
        return self.rewards

class ValueiterationAgent(CaptureAgent):
    #def __init__(self, gameState,env,discount = 0.9):
    def registerInitialState(self, gameState,discount = 0.9):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.discount=discount

    def chooseAction(self, gameState):
        self.env = Env(gameState)
        policies = valueiteration(self, iter=10)

        return policies[gameState.getAgentPosition(self.index)]

    def valueiteration(self, iter):
        for _ in range(iter):
            valueiterationAgent.update_values()
        #TODO: come up a policy map

        return policies

    def update_values(self):
        for i, j in self.env.poslist:

            self.env.rewards[i, j] = self.discount * max(self.successors_value(i, j))

    def successors_value(self, i, j):
        successors = self.env.getSuccessor(i, j)
        values = [self.env.rewards[i, j] for i, j in successors]
        return values

    def best_policy(self, i, j):
        successors = self.env.getSuccessor(i, j)
        best_state_index = np.argmax(self.successors_value(i, j))
        best_state_pos = successors[best_state_index]
        
        return self.best_policy(best_state[0], best_state[1])


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


