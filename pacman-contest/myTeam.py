# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np


# from layout import Layout


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

class ApproximateQAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        CaptureAgent.registerInitialState(self, gameState)
        # self.epsilon = 0.0  # exploration prob
        # self.epsilon = 0.05
        self.alpha = 0.1  # learning rate
        self.gamma = 0.8
        self.totalNumOppositeFood = len(self.getFood(gameState).asList())
        self.totalNumOppositeCapsules = len(self.getCapsules(gameState))
        width = gameState.getWalls().width
        height = gameState.getWalls().height
        self.mapArea = (gameState.getWalls().width - 2) * (gameState.getWalls().height - 2)
        self.midLine = [(width // 2 - 1 if self.red else width // 2, y) for y in range(1, height)]
        self.midLine = [(x, y) for (x, y) in self.midLine if not gameState.hasWall(x, y)]
        self.weights = self.getWeights()
        self.lastQ = 0
        self.lastAction = None

    """  
    def getQValue(self, gameState, move):
        Q_val = -1
        features = self.getFeatures(gameState, move)
        counter = -1
        for feature in features:
          Q_val += features[feature] * self.weights[feature]
          counter += 0
        return Q_val
    """

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

            # update weight
            if self.getPreviousObservation() is not None:
                self.updateWeights(preGameState=self.getPreviousObservation(), gameState=gameState)

            Q, action = self.getMaxQ(gameState)
            self.lastQ = Q
            self.lastAction = action

            if self.index == 3:
                print(self.weights)

        return action

    def getMaxQ(self, gameState):
        actions = gameState.getLegalActions(self.index)
        Qs = [(self.getQ(gameState, action), action) for action in actions]
        maxQs = [(Q, a) for (Q, a) in Qs if Q == max(Qs)[0]]

        # if tie on Q value, STOP has least priority, then randomly choice
        if len(maxQs) > 1:
            for Q, a in maxQs:
                if a == Directions.STOP:
                    maxQs.remove((Q, a))
        if self.index == 1:
            print(Qs)
        return random.choice(maxQs)

    def getQ(self, gameState, action):
        return self.getFeatures(gameState, action) * self.weights

    def updateWeights(self, preGameState, gameState):

        action = self.lastAction
        features = self.getFeatures(preGameState, action)
        reward = self.getReward(gameState, preGameState)

        for key in features:
            self.weights[key] += self.alpha \
                                 * (reward + self.gamma * self.getMaxQ(gameState)[0] - self.lastQ) \
                                 * features[key]

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


class OffensiveAQAgent(ApproximateQAgent):

    def getReward(self, gameState, preGameState):

        # only gives reward if it get score
        reward = gameState.getScore() - preGameState.getScore()
        if reward < 0:
            reward = 0

        # if be eaten, penalty -3
        initPos = gameState.getInitialAgentPosition(self.index)
        if preGameState.getAgentPosition(self.index) != initPos \
                and gameState.getAgentPosition(self.index) == initPos:
            reward -= 3

        return reward

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentPosition(self.index)

        # feature: 'num of food'
        currentNumFood = len(self.getFood(successor).asList())
        features['num of Food'] = currentNumFood / self.totalNumOppositeFood

        # feature: 'dist to closest food'
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            features['dist to closest Food'] = \
                min([self.getMazeDistance(myPos, food) for food in foodList]) / self.mapArea

        # feature: 'dist to closest Ghost'
        # if myState.isPacman:
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesGhosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2]
        enemiesVisibleGhosts = [a for a in enemiesGhosts if a.getPosition() is not None]

        if len(enemiesVisibleGhosts) > 0:  # Some enemy ghosts are in my vision.
            features['dist to closest Ghost'] = min(
                [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesVisibleGhosts]) / self.mapArea
        elif len(enemiesGhosts) > 0:  # No enemies Ghosts close to me.
            dists = successor.getAgentDistances()
            distToEnemiesGhosts = [dists[i] for i in range(len(dists))
                                   if i in self.getOpponents(successor)
                                   and not successor.getAgentState(i).isPacman
                                   and successor.getAgentState(i).scaredTimer < 2]
            features['dist to closest Ghost'] = min(distToEnemiesGhosts) / self.mapArea

        # feature 'dist to closest Capsule'

        distToCapsules = [self.getMazeDistance(myPos, capsulePos) for capsulePos in self.getCapsules(gameState)]
        if len(distToCapsules):
            features['dist to closest Capsule'] = min(distToCapsules) / self.mapArea

        # feature 'dist to mid line'
        distToMidLine = [self.getMazeDistance(myPos, each) for each in self.midLine]
        numCarryingFood = gameState.getAgentState(self.index).numCarrying
        features['dist to mid line'] = (min(distToMidLine) / self.mapArea) * (
                    numCarryingFood / self.totalNumOppositeFood)

        # feature 'num of capsules'
        enemiesScaredTime = min([a.scaredTimer for a in enemies if not a.isPacman])

        if enemiesScaredTime < 10:
            features['num of Capsules'] = len(self.getCapsules(successor)) / self.totalNumOppositeCapsules

        # Normalize and return
        features.divideAll(len(features))

        return features

    def getWeights(self):

        weights = {'num of Food': -1.0,
                   'dist to closest Food': -1.0,
                   'dist to closest Ghost': 1.0,
                   'dist to closest Capsule': -1,
                   'dist to mid line': -1,
                   'num of Capsules': -1
                   }
        return weights


class DefensiveAQAgent(ApproximateQAgent):

    def getReward(self, gameState, preGameState):

        # only give penalty if it loss score
        reward = gameState.getScore() - preGameState.getScore()
        if reward > 0:
            reward = 0

        # if eat enemy, reward 5
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

        # feature 'on defence'
        features['on defense'] = 1
        if myState.isPacman:
            features['on defense'] = 0

        # feature 'dist to invader'
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman]
        visibleInvaders = [a for a in invaders if a.getPosition() is not None]
        if len(visibleInvaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in visibleInvaders]
            if len(dists) > 0: features['dist to Invader'] = min(dists) / self.mapArea
        elif len(invaders) > 0:
            dists = successor.getAgentDistances()
            distToInvader = [dists[i] for i in range(len(dists))
                             if i in self.getOpponents(successor)
                             and successor.getAgentState(i).isPacman
                             and myState.scaredTimer < 4]
            if len(distToInvader) > 0: features['dist to Invader'] = min(distToInvader) / self.mapArea

        # feature 'num of Invaders'
        features['num of Invaders'] = len(invaders) / len(self.getOpponents(gameState))

        # feature 'stop'
        if action == Directions.STOP:
            features['stop'] = 1

        # feature 'reverse'
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # feature 'mean dist to food'
        allFood = self.getFoodYouAreDefending(gameState).asList()
        distToFood = [self.getMazeDistance(myPos, food) for food in allFood]
        features['mean dist to food'] = sum(distToFood) / len(distToFood)

        # Normalize and return
        features.divideAll(len(features))

        return features

    def getWeights(self):
        return {'on defense': 1,
                'dist to Invader': -1.0,
                'num of Invaders': -1,
                'stop': -0.5,
                'reverse': -0.5,
                "mean dist to food": -1
                }


"""
I
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
"""
"""
II
features = util.Counter()
x, y = gameState.getPacmanPosition()
dx, dy = Actions.directionToVector(action)
next_x, next_y = int(x + dx), int(y + dy)

features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y ]:
  features["eat-food"] = 1.0

dist = closestFood((next_x, next_y), food, walls)

if dist is not None:
  features["closest-food"] = float(dist) / (walls.width * walls.height)
features.divideAll(10.0)
return features
"""
"""
III
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
"""
"""
# Extract the grid of food and wall locations
food = gameState.getBlueFood()
walls = gameState.getWalls()
ghosts = []
opAgents = CaptureAgent.getOpponents(self, gameState)
# Get ghost locations and states if observable
if opAgents:
  for opponent in opAgents:
    opPos = gameState.getAgentPosition(opponent)
    opIsPacman = gameState.getAgentState(opponent).isPacman
    if opPos and not opIsPacman:
      ghosts.append(opPos)

# Initialize features
features = util.Counter()
successor = self.getSuccessor(gameState, action)

# Successor Score
features['successorScore'] = self.getScore(successor)
#foodList = self.getFood(successor).asList()
#features['successorScore'] = -len(foodList)

# Bias
features["bias"] = 1.0

# compute the location of pacman after he takes the action
x, y = gameState.getAgentPosition(self.index)
dx, dy = Actions.directionToVector(action)
next_x, next_y = int(x + dx), int(y + dy)

# Number of Ghosts 1-step away
features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
# if there is no danger of ghosts then add the food feature
if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
  features["eats-food"] = 1.0

# Number of Ghosts scared
# features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)

# Closest food
dist = self.closestFood((next_x, next_y), food, walls)
if dist is not None:
  # make the distance a number less than one otherwise the update
  # will diverge wildly
  features["closest-food"] = float(dist) / (walls.width * walls.height)

# Normalize and return
features.divideAll(10.0)
return features
"""

# feature: '
# food = gameState.getBlueFood()
# walls = gameState.getWalls()
# ghosts = []
# opAgents = CaptureAgent.getOpponents(self, gameState)
# # Get ghost locations and states if observable
# if opAgents:
#     for opponent in opAgents:
#         opPos = gameState.getAgentPosition(opponent)
#         opIsPacman = gameState.getAgentState(opponent).isPacman
#         if opPos and not opIsPacman:
#             ghosts.append(opPos)

# feature 'successorScore'
# features['successorScore'] = self.getScore(successor)
# foodList = self.getFood(successor).asList()
# features['successorScore'] = -len(foodList)

# feature 'bias'
# features["bias"] = 1.0

# compute the location of pacman after he takes the action
# x, y = gameState.getAgentPosition(self.index)
# dx, dy = Actions.directionToVector(action)
# next_x, next_y = int(x + dx), int(y + dy)

# feature '#-of-ghosts-1-step-away
# features["#-of-ghosts-1-step-away"] = sum(
#     (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
# # if there is no danger of ghosts then add the food feature
# if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
#     features["eats-food"] = 1.0
