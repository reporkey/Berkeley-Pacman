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
class ValueIteration:

    def __init__(self, gameState, index, epoch, heuristic, discount):
        self.index = index
        #self.epoch=epoch
        self.isRed = gameState.isOnRedTeam(index)
        self.discount = discount
        self.width, self.height = gameState.getWalls().width, gameState.getWalls().height
        self.rewards = np.zeros((self.width, self.height), dtype=None)
        self.Vs = np.zeros((self.width, self.height), dtype=None)
        self.policies = np.full((self.width, self.height), None)
        self.toUpdate = []

        self.buildVMap(gameState, heuristic)
        self.iteration(epoch)
        self.buildPoliciesMap()


    def buildVMap(self, gameState,heuristic):
        walls = gameState.getWalls().asList()
        foods = gameState.getBlueFood().asList() if self.isRed else gameState.getRedFood().asList()
        capsules = gameState.getBlueCapsules() if self.isRed else gameState.getRedCapsules()
        width, height = gameState.getWalls().width, gameState.getWalls().height
        numCarrying = gameState.getAgentState(self.index).numCarrying
        deliveryLine = [(self.width // 2 - 1 if self.isRed else self.width // 2, y) for y in range(1, self.height)]


        # build reward map
        for x in range(width):  # set out boundary cell to None
            self.rewards[x][0] = None
            self.rewards[x][-1] = None
        for y in range(height):  # set out boundary cell to None
            self.rewards[0][y] = None
            self.rewards[-1][y] = None

        # evaluate heuristically
        for (x, y) in foods:  # set reward of each food as 10
            self.rewards[x][y] += heuristic["food"]
        for (x, y) in capsules:  # set reward of each capsule as 100
            self.rewards[x][y] += heuristic["capsule"]
        for (x, y) in deliveryLine:
            self.rewards[x][y] += heuristic["delivery"] * numCarrying
        for (x, y) in walls:  # set reward of each WALL as None
            self.rewards[x][y] = None

        # lable visible enemies
        enemyIndices = gameState.getBlueTeamIndices() if self.isRed else gameState.getRedTeamIndices()
        for enemyIndex in enemyIndices:
            enemyState = gameState.getAgentState(enemyIndex)
            if enemyState.configuration is not None:
                x, y = enemyState.getPosition()
                # I'm Ghost, enemy is pacman
                if enemyState.isPacman:
                    # If I get scared
                    if gameState.getAgentState(self.index).scaredTimer > 0:
                        self.rewards[int(x)][int(y)] += heuristic["enemyGhost"]
                    else:
                        self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]
                # I'm pacman, enemy is ghost
                if not enemyState.isPacman and gameState.getAgentState(self.index).isPacman:
                    if enemyState.scaredTimer > 0:
                        self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]
                    else:
                        self.rewards[int(x)][int(y)] += (
                        heuristic["enemyGhost"] + heuristic["foodLostPenalty"] * numCarrying)

        # TODO: assign "food delivery" reward based on number of its eaten; also higher penalty on ghost if eaten more

        # list of position that required to be update during iterations
        self.toUpdate = [pos for pos, x in np.ndenumerate(self.rewards) if x == 0]
        # print(self.toUpdate)


    def iteration(self, epoch):
        self.Vs = self.rewards.copy()

      # update all V values [epoch] times
        for _ in range(epoch):
            oldVs = self.Vs.copy()
            for i, j in self.toUpdate:
                self.Vs[i, j] = self.discount * max(self.getSuccessors(oldVs, i, j).values())

    def buildPoliciesMap(self):
        # make up a policy map from V values
        for (x, y), value in np.ndenumerate(self.Vs):
            if not np.isnan(value):
                successors = self.getSuccessors(self.Vs, x, y)
                (i, j) = max(successors, key=successors.get)
                if (i - x, j - y) == (0, 1):
                    self.policies[x, y] = Directions.NORTH
                elif (i - x, j - y) == (0, -1):
                    self.policies[x, y] = Directions.SOUTH
                elif (i - x, j - y) == (1, 0):
                    self.policies[x, y] = Directions.EAST
                elif (i - x, j - y) == (-1, 0):
                    self.policies[x, y] = Directions.WEST
                elif (i - x, j - y) == (0, 0):
                    self.policies[x, y] = Directions.STOP
                else:
                    self.policies[x, y] = None
            else:
                self.policies[x, y] = None

    def getSuccessors(self, grid, i, j):
        successors = {}  # successor = {(x, y) = V_value}
        if i - 1 >= 0 and not np.isnan(grid[i - 1, j]):
            successors[(i - 1, j)] = grid[i - 1, j]
        if i + 1 <= self.width and not np.isnan(grid[i + 1, j]):
            successors[(i + 1, j)] = grid[i + 1, j]
        if j - 1 >= 0 and not np.isnan(grid[i, j - 1]):
            successors[(i, j - 1)] = grid[i, j - 1]
        if j + 1 <= self.height and not np.isnan(grid[i, j + 1]):
            successors[(i, j + 1)] = grid[i, j + 1]

        return successors


class ValueiterationAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        valueIteration=ValueIteration(gameState, self.index, 100, self.getHeuristic(),0.9)
        (x,y)=gameState.getAgentPosition(self.index)
        return valueIteration.policies[x,y]

    def getHeuristic(self):

        """
        overwrite by subclass
        """

        features = util.Counter()
        features['food'] = 1
        features['capsule'] = 1
        features['enemyGhost'] = -1
        features['enemyPacman'] = 1
        return features


class OffensiveReflexAgent(ValueiterationAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getHeuristic(self):
      features = util.Counter()
      features['food'] = 100
      features['capsule'] = 200
      features['delivery'] = 30
      features['foodLostPenalty'] = -100
      features['enemyGhost'] = -10000
      features['enemyPacman'] = 200
      return features


class DefensiveReflexAgent(ValueiterationAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def getHeuristic(self):
      features = util.Counter()
      features['food'] = 100
      features['capsule'] = 0
      features['delivery'] = 20
      features['foodLostPenalty'] = -100
      features['enemyGhost'] = -10000
      features['enemyPacman'] = 50000
      return features
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
"""





