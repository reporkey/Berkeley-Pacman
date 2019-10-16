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


class ValueIterationAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        self.width, self.height = gameState.getWalls().width, gameState.getWalls().height
        self.discount = 0.9
        self.timeLimit = 0.4
        self.heuristic = self.getHeuristic()

        self.lastDefending = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
        self.enemiesPos = [gameState.getAgentPosition(i)
                           if i in self.getOpponents(gameState) else None
                           for i in range(4)]

    def chooseAction(self, gameState):

        # update enemy position info
        self.updateEnemiesPos(gameState)

        # define variable
        self.start = time.time()
        self.rewards = np.zeros((self.width, self.height), dtype=None)
        self.Vs = np.zeros((self.width, self.height), dtype=None)
        self.policies = np.full((self.width, self.height), None)
        self.toUpdate = []

        # value iteration
        self.buildVMap(gameState, self.getHeuristic())
        self.iteration(self.timeLimit - 0.1)
        self.buildPoliciesMap()

        (x, y) = gameState.getAgentPosition(self.index)

        return self.policies[x, y]

    def buildVMap(self, gameState, heuristic):
        walls = gameState.getWalls().asList()
        food = self.getFood(gameState).asList()
        defendingFood = self.getFoodYouAreDefending(gameState).asList()
        capsules = self.getCapsules(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying
        deliveryLine = [(self.width // 2 - 1 if self.red else self.width // 2, y) for y in range(1, self.height)]

        # build reward map
        for x in range(self.width):  # set out boundary cell to None
            self.rewards[x][0] = None
            self.rewards[x][-1] = None
        for y in range(self.height):  # set out boundary cell to None
            self.rewards[0][y] = None
            self.rewards[-1][y] = None

        for (x, y) in food:
            if len(food) <= 2:
                self.rewards[x][y] +=0
            else:
                self.rewards[x][y] += heuristic["food"]

        for (x, y) in capsules:
            self.rewards[x][y] += heuristic["capsule"]

        for (x, y) in deliveryLine:
            self.rewards[x][y] += heuristic["delivery"] * numCarrying

        # label visible enemies
        for index, pos in enumerate(self.enemiesPos):
            if index in self.getOpponents(gameState) and pos is not None:

                x, y = pos
                enemyState = gameState.getAgentState(index)

                # Enemy is pacman
                if enemyState.isPacman:

                    # I'm scared
                    if gameState.getAgentState(self.index).scaredTimer > 0:
                        self.rewards[int(x)][int(y)] += heuristic["enemyGhost"]

                    # I'm not scared
                    else:
                        self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]

                # I'm pacman, enemy is ghost
                elif gameState.getAgentState(self.index).isPacman and not enemyState.isPacman:

                    # Enemy is scared
                    # if enemyState.scaredTimer > 0:
                    #     self.rewards[int(x)][int(y)] += heuristic["enemyPacman"]
                    #

                    # Enemy is not scared
                    if enemyState.scaredTimer < 3:
                        self.rewards[int(x)][int(y)] += (heuristic["enemyGhost"] + heuristic["foodLostPenalty"] * numCarrying)

        for (x, y) in walls:
            self.rewards[x][y] = None

        self.toUpdate = [pos for pos, x in np.ndenumerate(self.rewards) if x == 0]

    def iteration(self, timeLimit):

        self.Vs = self.rewards.copy()

        while time.time() - self.start < timeLimit:

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

    def updateEnemiesPos(self, gameState):

        # predict pos by last eaten food/capsule
        dist = lambda pos: self.getMazeDistance(gameState.getAgentPosition(self.index), pos)
        enemiesDists = [dist(self.enemiesPos[i])
                        if i in self.getOpponents(gameState)
                           and self.enemiesPos[i] is not None
                        else None
                        for i in range(4)]
        if enemiesDists.count(not None) > 0:
            closedEnemyIndex = enemiesDists.index(min(x for x in enemiesDists if x is not None))
        else:
            closedEnemyIndex = self.getOpponents(gameState)[0]
        defending = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
        for each in self.lastDefending:
            if each not in defending:
                self.enemiesPos[closedEnemyIndex] = each
        self.lastDefending = defending

        # observe pos by own vision
        for i in self.getOpponents(gameState):
            if gameState.getAgentState(i).getPosition() is not None:
                self.enemiesPos[i] = gameState.getAgentState(i).getPosition()

        # remove position of lost enemy when refreshing fog of war
        myPos = gameState.getAgentPosition(self.index)
        visibleEnemiesPos = []
        for i in range(4):
            if i in self.getOpponents(gameState):
                if gameState.getAgentState(i).getPosition() is not None:
                    visibleEnemiesPos.append(gameState.getAgentPosition(i))
                else:
                    visibleEnemiesPos.append(None)
            else:
                visibleEnemiesPos.append(None)
        for i in self.getOpponents(gameState):
            if self.enemiesPos[i] is not None \
                    and distanceCalculator.manhattanDistance(myPos, self.enemiesPos[i]) < 5 \
                    and self.enemiesPos[i] not in visibleEnemiesPos:
                self.enemiesPos[i] = None

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
        features['delivery'] = 20
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -10000
        features['enemyPacman'] = 50
        return features


class DefensiveVIAgent(ValueIterationAgent):

    def getHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 0
        features['delivery'] = 40
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -1000
        features['enemyPacman'] = 100000
        return features
