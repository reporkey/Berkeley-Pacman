from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np
from game import Grid


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveVIAgent', second='DefensiveVIAgent'):
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

    def __init__(self, gameState, index, heuristic, discount, epoch=500, timeLimit=0.9):
        self.index = index
        self.epoch=epoch
        self.timeLimit = timeLimit
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

    def buildVMap(self, gameState, heuristic):
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

        for (x, y) in foods:
            self.rewards[x][y] += heuristic["food"]
        for (x, y) in capsules:
            self.rewards[x][y] += heuristic["capsule"]
        for (x, y) in deliveryLine:
            self.rewards[x][y] += heuristic["delivery"] * numCarrying
        for (x, y) in walls:  # set reward of each WALL as None
            self.rewards[x][y] = None

        # label visible enemies
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
                        self.rewards[int(x)][int(y)] += \
                            (heuristic["enemyGhost"] + heuristic["foodLostPenalty"] * numCarrying)

        # list of position that required to be update during iterations
        self.toUpdate = [pos for pos, x in np.ndenumerate(self.rewards) if x == 0]

    def iteration(self, epoch):
        self.Vs = self.rewards.copy()

        # update all V values [epoch] times
        for _ in range(epoch):
            oldVs = self.Vs.copy()
            for i, j in self.toUpdate:
                self.Vs[i, j] = self.discount * max(self.getSuccessors(oldVs, i, j).values())
            if time.time() - self.start > self.timeLimit - 0.05:
                break

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


class ValueIterationAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.gamma = 0.9
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        valueIteration = ValueIteration(gameState=gameState, index=self.index, discount=self.gamma,
                                        heuristic=self.getHeuristic(), timeLimit=0.9)
        (x, y) = gameState.getAgentPosition(self.index)
        return valueIteration.policies[x, y]

    def getHeuristic(self):
        """
        overwrite by subclass
        """

        features = util.Counter()
        return features


class OffensiveVIAgent(ValueIterationAgent):

    def getHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 200
        features['delivery'] = 30
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -10000
        features['enemyPacman'] = 200
        return features


class DefensiveVIAgent(ValueIterationAgent):

    def getHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 0
        features['delivery'] = 20
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -10000
        features['enemyPacman'] = 50000
        return features
