from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np
import os, json

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAQAgent', second='DefensiveAQAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

ALPHA = 0.01  # learning rate
GAMMA = 0.8
NSTEP = 3

class ApproximateQAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)
        self.totalNumOppositeFood = len(self.getFood(gameState).asList())
        self.totalNumOppositeCapsules = len(self.getCapsules(gameState))
        self.lastDefending = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(
            gameState)
        self.enemiesPos = [gameState.getAgentPosition(i) if i in self.getOpponents(gameState) else None for i in
                           range(4)]
        width = gameState.getWalls().width
        height = gameState.getWalls().height
        self.mapArea = (gameState.getWalls().width - 2) * (gameState.getWalls().height - 2)
        self.midLine = [(width // 2 - 1 if self.red else width // 2, y) for y in range(1, height)]
        self.midLine = [(x, y) for (x, y) in self.midLine if not gameState.hasWall(x, y)]
        self.enemyMidLine = [(width // 2 if self.red else width // 2 - 1, y) for y in range(1, height)]
        self.enemyMidLine = [(x, y) for (x, y) in self.midLine if not gameState.hasWall(x, y)]

        self.lastAction = None
        self.lastQ = 0
        self.lastFeatures = util.Counter()

        self.file = ""
        self.weights = self.getWeights()

    def chooseAction(self, gameState):

        legalActions = gameState.getLegalActions(self.index)
        action = None

        if len(legalActions) > 0:
            """
            prob = util.flipCoin(self.epsilon)
            if prob:
              action = random.choice(legalActions)
            else:
            """

            # update enemy position info
            self.updateEnemiesPos(gameState)

            start = time.time()
            action, Q = self.getMaxQ(gameState, step=NSTEP)
            feature = self.getFeatures(gameState, action)

            # update weight, except 1st time step
            if self.getPreviousObservation() is not None and self.lastAction is not None:
                self.updateWeights(preGameState=self.getPreviousObservation(), gameState=gameState, Q=Q)

            self.lastQ = Q
            self.lastAction = action
            self.lastFeatures = feature

        return action

    def final(self, gameState):

        # update weights
        self.updateWeights(preGameState=self.getPreviousObservation(), gameState=gameState, Q=self.lastQ/GAMMA)

        # write weights into a file
        file = open(self.file, 'a')
        data = json.dumps(self.weights)
        file.write(data + '\n')
        file.close()

    def getMaxQ(self, gameState, step):

        actions = gameState.getLegalActions(self.index)
        Qs = []
        for action in actions:
            if step > 1:
                successor = self.getSuccessor(gameState, action)
                a, q = self.getMaxQ(successor, step-1)
                Q = self.getFeatures(gameState, action) * self.weights
                Qs.append(Q + GAMMA * (q - Q))
            else:
                Qs.append(self.getFeatures(gameState, action) * self.weights)

        if len(Qs) > 1:
            bestAction = actions[np.argmax(Qs)]
            bestQ = np.max(Qs)
        else:
            print("No Features Found for All Actions.", "Index:", self.index, gameState.getAgentPosition(self.index), actions)
            return random.choice(actions), None

        return bestAction, bestQ

    def updateWeights(self, preGameState, gameState, Q):

        # try:

        reward = self.getReward(gameState, preGameState)
        for feature in self.lastFeatures:
            self.weights[feature] += ALPHA \
                                 * (reward + GAMMA * Q - self.lastQ) \
                                 * self.lastFeatures[feature]
            if self.weights[feature] < 0:
                self.weights[feature] = 0

        # except:  # for the finish update, sometimes pre gs and gs are not consistent
        #     pass

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

    def updateEnemiesPos(self, gameState):

        myPos = gameState.getAgentPosition(self.index)
        defending = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

        # predict pos by last eaten food/capsule
        enemiesDists = [self.getMazeDistance(myPos, self.enemiesPos[i])
                        if i in self.getOpponents(gameState)
                           and self.enemiesPos[i] is not None
                        else np.nan
                        for i in range(4)]
        if enemiesDists.count(not np.nan) > 0:
            closedEnemyIndex = np.nanargmin(enemiesDists)
        else:
            closedEnemyIndex = self.getOpponents(gameState)[0]
        for each in self.lastDefending:
            if each not in defending:
                self.enemiesPos[closedEnemyIndex] = each
        self.lastDefending = defending

        # observe pos by own vision
        for i in self.getOpponents(gameState):
            if gameState.getAgentState(i).getPosition() is not None:
                self.enemiesPos[i] = gameState.getAgentState(i).getPosition()

        # remove position of enemy that lost tracking when re-explores fog of war
        visibleEnemiesPos = [gameState.getAgentState(i).getPosition()
                             if i in self.getOpponents(gameState)
                             else None
                             for i in range(4)]
        for i in self.getOpponents(gameState):
            if self.enemiesPos[i] is not None \
                    and distanceCalculator.manhattanDistance(myPos, self.enemiesPos[i]) < 5 \
                    and self.enemiesPos[i] not in visibleEnemiesPos:
                self.enemiesPos[i] = None

    def getReward(self, gameState, preGameState):
        """
        Overwrite
        """
        return 0

    def getFeatures(self, gameState, action):
        """
        Overwrite
        """
        features = util.Counter()
        return features

    def getWeights(self):
        """
        Overwrite
        """
        weights = util.Counter()
        return weights


class OffensiveAQAgent(ApproximateQAgent):

    def getReward(self, gameState, preGameState):

        # successfully delivery food (only count positive score)
        reward = 3 * (gameState.getScore() - preGameState.getScore())
        reward = reward if self.red else -reward
        if reward < 0:
            reward = 0

        foodChange = len(self.getFood(preGameState).asList()) - len(self.getFood(gameState).asList())
        capsuleChange = len(self.getCapsules(preGameState)) - len(self.getCapsules(gameState))

        # eat food
        if foodChange > 0:
            reward += 0.3

        # lose food
        if foodChange < 0:
            reward -= foodChange

        # eat Capsule
        if capsuleChange > 0:
            reward += 0.5

        # do nothing
        if capsuleChange == 0 and foodChange == 0:
            reward -= 0.05

        # be eaten
        initPos = gameState.getInitialAgentPosition(self.index)
        if preGameState.getAgentPosition(self.index) != initPos \
                and gameState.getAgentPosition(self.index) == initPos:
            reward -= 5

        return reward

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentPosition(self.index)
        numCarryingFood = successor.getAgentState(self.index).numCarrying
        foodList = self.getFood(successor).asList()

        # feature: 'num of food'
        # feature: 'dist to closest food'
        if len(foodList) > 2:
            features['num of food'] = - len(foodList) / self.totalNumOppositeFood
            features['dist to closest food'] = - min(
                [self.getMazeDistance(myPos, food) for food in foodList]) / self.mapArea

        # feature: 'dist to closest ghost'
        distsToGhosts = [self.getMazeDistance(myPos, pos)
                         for index, pos in enumerate(self.enemiesPos)
                         if index in self.getOpponents(successor)
                         and pos is not None
                         and not successor.getAgentState(index).isPacman
                         and successor.getAgentState(index).scaredTimer < 3]
        if len(distsToGhosts) > 0:
            features['dist to closest ghost'] = ((10 + numCarryingFood) / self.totalNumOppositeFood) \
                                                * min(distsToGhosts) / self.mapArea

        # feature 'dist to closest capsule'
        # feature 'num of capsules'
        flag = [True if successor.getAgentState(index).scaredTimer < dist < 10 else False
                for index, dist in enumerate(successor.getAgentDistances())
                if index in self.getOpponents(successor)]
        if True in flag:
            distToCapsules = [self.getMazeDistance(myPos, pos) for pos in self.getCapsules(successor)]
            if len(distToCapsules) > 0:
                features['dist to closest capsule'] = - min(distToCapsules) / self.mapArea
                features['num of capsules'] = - len(self.getCapsules(successor)) / self.totalNumOppositeCapsules

        # feature 'dist to mid line'
        distToMidLine = [self.getMazeDistance(myPos, each) for each in self.midLine]
        features['dist to mid line'] = - (min(distToMidLine) / self.mapArea) \
                                       * (numCarryingFood / self.totalNumOppositeFood)

        # Normalize and return
        features.divideAll(len(self.weights))

        return features

    def getWeights(self):

        # self.file = os.path.join(os.path.dirname(__file__), "offensive.json")

        sys.path.append('teams/Pacman_Go/')
        self.file = "./offensive.json"

        f = open(self.file, "r")
        data = f.read().splitlines()[-1]
        weights = json.loads(data)
        f.close()

        return weights


class DefensiveAQAgent(ApproximateQAgent):

    def getReward(self, gameState, preGameState):

        # enemy successfully delivery food (only count negative score)
        reward = gameState.getScore() - preGameState.getScore()
        reward = reward if self.red else -reward
        if reward > 0:
            reward = 0

        foodChange = len(self.getFoodYouAreDefending(gameState).asList()) - len(
            self.getFoodYouAreDefending(preGameState).asList())
        capsuleChange = len(self.getCapsulesYouAreDefending(gameState)) - len(
            self.getCapsulesYouAreDefending(preGameState))

        # enemy eats food
        if foodChange < 0:
            reward -= 1

        # enemy loses food
        if foodChange > 0:
            reward += foodChange

        # enemy eats Capsule
        if capsuleChange < 0:
            reward -= 2

        # eat enemy
        myPos = gameState.getAgentPosition(self.index)
        enemies = [preGameState.getAgentState(i) for i in self.getOpponents(preGameState)]
        enemiesPrePos = [e.getPosition for e in enemies if e.getPosition() is not None]
        if myPos in enemiesPrePos:
            reward += 5

        return reward

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # feature 'dist to invader'
        distsToInvaders = [self.getMazeDistance(myPos, pos)
                           for index, pos in enumerate(self.enemiesPos)
                           if index in self.getOpponents(successor)
                           and pos is not None
                           and successor.getAgentState(index).isPacman]
        if myState.scaredTimer > 0:
            distsToInvaders = [abs(dist - 2) for dist in distsToInvaders]
        if len(distsToInvaders) > 0:
            features['dist to invader'] = - min(distsToInvaders) / self.mapArea

        # feature 'num of Invaders'
        invaders = [successor.getAgentState(i).isPacman for i in self.getOpponents(successor)]
        features['num of invaders'] = - len(invaders) / len(self.getOpponents(gameState))

        # feature 'mean dist to food'
        allFood = self.getFoodYouAreDefending(gameState).asList()
        distToFood = [self.getMazeDistance(myPos, food) for food in allFood]
        if len(distToFood) > 0:
            features['mean dist to food'] = - sum(distToFood) / len(distToFood) / self.mapArea

        # feature 'dist to food that closed to mid line'
        midLineToFood = []
        for food in allFood:
            dist = [self.getMazeDistance(food, pos) for pos in self.enemyMidLine]
            if len(dist) > 0:
                midLineToFood.append(min(dist))
            else:
                midLineToFood.append(np.nan)
        if midLineToFood.count(np.nan) < len(midLineToFood):
            foodIndex = np.nanargmin(midLineToFood)
            features['dist to food that closed to mid line'] \
                = - self.getMazeDistance(myPos, allFood[foodIndex]) / self.mapArea

        # Normalize and return
        features.divideAll(len(self.weights))

        return features

    def getWeights(self):

        # self.file = os.path.join(os.path.dirname(__file__), "defensive.json")

        # sys.path.append('teams/Pacman_Go/')
        # self.file = "./defensive.json"

        f = open(self.file, "r")
        data = f.read().splitlines()[-1]
        weights = json.loads(data)
        f.close()

        return weights
