from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np
from game import Grid

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

class ValueIteration:

    def __init__(self, gameState, index, epoch, discount=0.9):
        self.index = index
        self.isRed = gameState.isOnRedTeam(index)
        self.discount = discount
        self.width, self.height = gameState.getWalls().width, gameState.getWalls().height
        self.rewards = np.zeros((self.width + 1, self.height + 1), dtype=None)
        self.Vs = np.zeros((self.width + 1, self.height + 1), dtype=None)
        self.policies = np.full((self.width + 1, self.height + 1), None)
        self.toUpdate = []

        self.buildVMap(gameState)
        self.iteration(epoch)
        self.buildPoliciesMap()

    def buildVMap(self, gameState):
        walls = gameState.getWalls().asList()
        foods = gameState.getBlueFood().asList() if self.isRed else gameState.getRedFood().asList()
        capsules = gameState.getBlueCapsules() if self.isRed else gameState.getRedCapsules()#.asList()
        width, height = gameState.getWalls().width, gameState.getWalls().height

        # build reward map
        for x in range(width+1):            # set out boundary cell to None
            self.rewards[x][0] = None
        for y in range(height+1):           # set out boundary cell to None
            self.rewards[0][y] = None
        for (x, y) in foods:                # set reward of each food as 10
            self.rewards[x][y] = 10
        for (x, y) in capsules:             # set reward of each capsule as 100
            self.rewards[x][y] = 100
        for (x, y) in walls:                # set reward of each WALL as None
            self.rewards[x][y] = None
        # print(self.rewards)

        # list of position that required to be update during iterations
        self.toUpdate = [pos for pos, x in np.ndenumerate(self.rewards) if x == 0]
        # print(self.toUpdate)

    def iteration(self, epoch):

        self.Vs = self.rewards.copy()

        # update all V values [epoch] times
        for n in range(epoch):
            oldVs = self.Vs.copy()
            for i, j in self.toUpdate:
                self.Vs[i, j] = self.discount * max(self.getSuccessors(oldVs, i, j).values())

    def buildPoliciesMap(self):
        # make up a policy map from V values
        for (x, y), value in np.ndenumerate(self.Vs):
            if not np.isnan(value):
                successors = self.getSuccessors(self.Vs, x, y)
                (i, j) = max(successors, key=successors.get)
                if (i-x, j-y) == (0, 1):
                    self.policies[x, y] = Directions.NORTH
                elif (i-x, j-y) == (0, -1):
                    self.policies[x, y] = Directions.SOUTH
                elif (i-x, j-y) == (1, 0):
                    self.policies[x, y] = Directions.EAST
                elif (i-x, j-y) == (-1, 0):
                    self.policies[x, y] = Directions.WEST
                elif (i-x, j-y) == (0, 0):
                    self.policies[x, y] = Directions.STOP
                else:
                    self.policies[x, y] = None
            else:
                self.policies[x, y] = None

    def getSuccessors(self, grid, i, j):
        successors = {}                 # successor = {(x, y) = V_value}
        if i-1 >= 0 and not np.isnan(grid[i-1, j]):
            successors[(i-1, j)] = grid[i-1, j]
        if i+1 <= self.width and not np.isnan(grid[i+1, j]):
            successors[(i+1, j)] = grid[i+1, j]
        if j-1 >= 0 and not np.isnan(grid[i, j-1]):
            successors[(i, j-1)] = grid[i, j-1]
        if j+1 <= self.height and not np.isnan(grid[i, j + 1]):
            successors[(i, j+1)] = grid[i, j+1]
        return successors

class ValueiterationAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        valueIteration=ValueIteration( gameState, self.index, epoch=100, discount=0.9)
        (x,y)=gameState.getAgentPosition(self.index)
        return valueIteration.policies[x,y]

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
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
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


