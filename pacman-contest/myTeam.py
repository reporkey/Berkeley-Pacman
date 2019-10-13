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

    def __init__(self, gameState, index, epoch, heuristic, discount):
        self.index = index
        self.start = time.time()
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
        selffoods=gameState.getRedFood().asList() if self.isRed else gameState.getBlueFood().asList()
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
        for (x, y) in selffoods:  # set reward of each food as 10
            if len(selffoods)<=8:
              self.rewards[x][y] += heuristic["selffood"]
            else:
              self.rewards[x][y] =0
        for (x, y) in capsules:  # set reward of each capsule as 100
            self.rewards[x][y] += heuristic["capsule"]
        for (x, y) in deliveryLine:
            if len(foods)<=2:
                self.rewards[x][y] += 5000 + heuristic["delivery"] * numCarrying #18 foods is enough, so  get back asap
            else:
                self.rewards[x][y] += heuristic["delivery"] * numCarrying
        for (x, y) in walls:  # set reward of each WALL as None
            self.rewards[x][y] = None

        # lable visible enemies
        enemyIndices = gameState.getBlueTeamIndices() if self.isRed else gameState.getRedTeamIndices()
        for enemyIndex in enemyIndices:
            enemyState = gameState.getAgentState(enemyIndex)
            if enemyState.configuration is not None:
                x, y = enemyState.getPosition()
                if enemyState.isPacman:
                    if gameState.getAgentState(self.index).scaredTimer > 0:
                        self.rewards[int(x)][int(y)] += heuristic["enemyGhost"]
                        #print(self.rewards[int(x)][int(y)])
                    else:
                        self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]
                # I'm pacman, enemy is ghost
                elif gameState.getAgentState(self.index).isPacman and not enemyState.isPacman:
                    if enemyState.scaredTimer > 0:
                        self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]
                    else:
                        self.rewards[int(x)][int(y)] += (
                        heuristic["enemyGhost"] + heuristic["foodLostPenalty"] * numCarrying)
                # I'm ghost, enemy is ghost
                else:
                    self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]

        # TODO: assign "food delivery" reward based on number of its eaten; also higher penalty on ghost if eaten more

        # list of position that required to be update during iterations
        self.toUpdate = [pos for pos, x in np.ndenumerate(self.rewards) if x == 0]
        # print(self.toUpdate)


    def iteration(self, epoch):
        self.Vs = self.rewards.copy()

      # update all V values [epoch] times
        # for _ in range(epoch): 
        while time.time() - self.start < 0.8:
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
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        valueIteration = ValueIteration(gameState, self.index, None, self.getHeuristic(), 0.9)
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
      features['selffood'] =0
      features['capsule'] = 200
      features['delivery'] = 20
      features['foodLostPenalty'] = -100
      features['enemyGhost'] = -10000
      features['enemyPacman'] = 50
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
      features['selffood'] = 20000
      features['capsule'] = 0
      features['delivery'] = 40
      features['foodLostPenalty'] = -100
      features['enemyGhost'] = -1000
      features['enemyPacman'] = 50000
      return features






