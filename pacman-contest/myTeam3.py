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

class Environment:
    def __init__(self, gameState):
        self.walls = gameState.getWalls()                              #walls是一个真值列表 wallsposition才是墙的坐标
        self.foods = self.getFood(gameState).asList()                  #food和walls的获取需要用不同的方法。可能跟他其他文件的相关定义有关
        self.capsules = self.getCapsules(gameState)
        #print(self.foods)
        width, height = self.walls.width, self.walls.height
        #print(width,height)
        wallsposition=[]                                                                   #the location of walls
        for x in range(width):
            for y in range(height):
                if self.walls[x][y]==True:
                    wallsposition.append([x,y])
        # print(wallspostion)

        self.rewards = np.full((height, width), 0)  # set every initial-reward 0. It also means the cost of move is 0
        for i in self.foods:  # set reward of each food as 10
            for x in range(width):
                for y in range(height):
                    if (x, y) == i:
                        self.rewards[height - y][x] = 10  # set reward of each food as 10
        for i in self.capsules:  # set reward of each capsule as 100
            for x in range(width):
                for y in range(height):
                    if (x, y) == i:
                        self.rewards[height - y][x] = 100

    def getReward(self, state, action, nextState):
        return self.rewards[nextState]-self.rewards[state]

    def getPossibleActions(self, state):
        return state.getLegalActions()

    def getStates(self):
        return self.rewards()


class valueiterationAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, environment, discount = 0.9, iterations = 100):
        self.environment=environment
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()   # A Counter is a dict with default 0
        states = environment.getStates()
        for i in range(iterations):
            valuesCopy = self.values.copy()
            for state in states:
                finalValue = None
                for action in state.getLegalActions():
                    currentValue = self.computeValue(state, action)
                    if finalValue == None or finalValue < currentValue:
                        finalValue = currentValue
                if finalValue == None:
                    finalValue = 0
                valuesCopy[state] = finalValue
            self.values = valuesCopy

    def computeValue(self, state, action):
        value=0
        successors=self.getSuccessor(state,action)
        for nextState in successors:
            value = self.evaluate(state,action,nextState)+ self.discount*self.values[nextState] #not value+=
        return value

    def choosemaxvalueAction(self, state):
        actions = self.environment.getPossibleActions(state)
        values=None
        result=None
        for action in actions:
          tempvalues = self.computeValue(state, action)
          if tempvalues >values:
            values=tempvalues
            result= action
        return result

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

    def evaluate(self, gameState, action):                                  #get r(s,a,s')
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
        for i in self.foods:                                                  #set reward of each food as 10
            for x in range(self.width):
                for y in range(self.height):
                    if (x,y)==i:
                        features[self.height-y][x]=10
        for i in self.capsules:                                              #set reward of each capsule as 100
            for x in range(self.width):
                for y in range(self.height):
                    if (x,y)==i:
                        features[self.height-y][x]=100
        print(features)
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(valueiterationAgent):
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


class DefensiveReflexAgent(valueiterationAgent):
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


