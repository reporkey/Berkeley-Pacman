"""
Since the game is not in prefect knowledge, the positions of enemy agent are not given, Monte Carlo tree search do not
able to expand or simulate enemies agent. Therefore, we apply "delete relaxing" from classical planning to this MDP,
the idea is "not concerning enemies", neither expanding or simulating. It is reasonable in uninformed path searching,
and should works pretty good in our situation. The game rules that when enemy agent is in 5 unit distance from my agent
position, then the position of that enemy agent will be given. Otherwise (manhattan distance > 5), rather than a
coordination, a noised maze distance is given. From the view of offensive agent, when enemy is far away from my
position (d > 5), its position may not as important as the position of foods.

Not sure about the performance on defensive agent, a reinforcement learning may outweigh.

Then it rise up another issue, how to treat my other agents. The current solution is to keep expanding and simulating.

"""

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
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


ALPHA = 0.03  # learning rate
GAMMA = 1  # discounted rate
C_P = 1.0  # UCT
EPSILON = 0.1  # e-greedy
TIME_LIMIT = 0.2


##########
# Agents #
##########

class MonteCarloAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

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

        self.file = ''
        self.weights = self.getWeights()

    def chooseAction(self, gameState):

        self.startTime = time.time()
        self.actions = gameState.getLegalActions(self.index)
        self.featuresOfActions = {action: util.Counter() for action in self.actions}
        self.root = Tree(index=self.index, state=gameState, action=None, r=None, depth=0)
        self.visibleAgents = [gameState.getAgentState(i)
                              if gameState.getAgentState(i).getPosition() is not None
                              else None
                              for i in range(4)]

        # update enemy position info
        self.updateEnemiesPos(gameState)

        # MCT search for best estimate
        self.MCTSearch(gameState)
        maxIndex = np.argmax([GAMMA * child.V for child in self.root.children])
        decision = self.root.children[int(maxIndex)].action

        Qs = [child.V for child in self.root.children]
        Q = np.max(Qs)
        action = self.root.children[np.argmax(Qs)].action
        feature = self.featuresOfActions[action]

        # update weight, except 1st time step
        if self.getPreviousObservation() is not None and self.lastAction is not None:
            self.updateWeights(preGameState=self.getPreviousObservation(), gameState=gameState, Q=Q)

        self.lastQ = Q
        self.lastAction = action
        self.lastFeatures = feature

        return action

    def MCTSearch(self, gameState):

        n = 0

        while True:
            n = n + 1
            selectNode, selectAction = self.selectNode()
            node = self.expandNode(selectNode, selectAction)
            V = self.simulate(node)
            self.backprop(V, node)
            if time.time() - self.startTime > TIME_LIMIT:
                break
        maxIndex = np.argmax([GAMMA * child.V for child in self.root.children])
        decision = self.root.children[int(maxIndex)].action
        for child in self.root.children:
            if child.action == Directions.NORTH:
                print("NORTH:\t", child.N, "\t", child.V)
            elif child.action == Directions.SOUTH:
                print("SOUTH:\t", child.N, "\t", child.V)
            elif child.action == Directions.WEST:
                print("WEST:\t", child.N, "\t", child.V)
            elif child.action == Directions.EAST:
                print("EAST:\t", child.N, "\t", child.V)
            elif child.action == Directions.STOP:
                print("STOP:\t", child.N, "\t", child.V)

        print("total: ", n, self.root.state.getAgentPosition(self.root.index), self.root.index, decision, "\n")

    def selectNode(self):
        node = self.root
        action = None
        actions = node.state.getLegalActions(node.index)

        # while (len(node.children) > 0 or len(actions) > 0) and not node.state.isOver():
        #
        #     # choose action
        #     existedActions = set([each.action for each in node.children])
        #     actions = list(set(actions) - existedActions)
        #     if len(actions) > 0:
        #         action = np.random.choice(actions)
        #         break

            # """ UCB1 (UCT) """
            # """ Since the Q is not normalized between attacker and defender, UCT is not a good choice """
            # UCB1 = np.array([child.V + 2 * C_P * np.sqrt(2 * np.log(node.N) / child.N) for child in node.children])
            # node = node.children[np.random.choice(np.flatnonzero(UCB1 == UCB1.max()))]
            # actions = node.state.getLegalActions(node.index)

        while action is None:
            """ backup """
            """ e-greedy """
            node = self.root
            actions = node.state.getLegalActions(node.index)

            while (len(node.children) > 0 or len(actions) > 0) and not node.state.isOver():

                existedActions = set([each.action for each in node.children])
                actions = list(set(actions) - existedActions)
                if len(actions) > 0:
                    action = np.random.choice(actions)
                    break

                if np.random.random() < EPSILON:
                    node = np.random.choice(node.children)
                else:
                    Qvalues = np.array([child.V for child in node.children])
                    node = node.children[np.random.choice(np.flatnonzero(Qvalues == Qvalues.max()))]
                actions = node.state.getLegalActions(node.index)

        # print(node.state.getAgentPosition(node.index), node.index, actions, "=>", action)
        return node, action

    def expandNode(self, parent, action):
        successorState = getSuccessor(parent.state, parent.index, action)
        successor = Tree(index=self.nextIndex(parent.index),
                         state=successorState,
                         action=action,
                         depth=parent.depth + 1,
                         r=None,
                         parent=parent)
        parent.children.append(successor)
        # print("expand Node from", parent.state.getAgentPosition(parent.index), action, "to ", successor.state.getAgentPosition(parent.index), "index: ", parent.index)
        return successor

    def simulate(self, node):
        features = self.getFeatures(node.state)
        Q = features * self.weights
        Q = Q if node.parent.index in self.getTeam(node.state) else -Q
        if node.depth == 1:
            self.featuresOfActions[node.action] = features
        return Q

    def backprop(self, V, successor):
        successor.V = V  # discounted future reward
        successor.N += 1

        node = successor.parent
        while node is not None:
            node.N += 1

            node.V = np.sum([node.V] + [(GAMMA * child.V) * child.N for child in node.children]) / node.N

            # bestChild = node.getBestChild()

            node = node.parent

    def final(self, gameState):

        # update weights
        self.updateWeights(preGameState=self.getPreviousObservation(), gameState=gameState, Q=self.lastQ / GAMMA)

        # write weights into a file
        file = open(self.file, 'a')
        data = json.dumps(self.weights)
        file.write(data + '\n')
        file.close()

    def updateWeights(self, preGameState, gameState, Q):

        reward = self.getReward(gameState, preGameState)
        for feature in self.lastFeatures:
            self.weights[feature] += ALPHA \
                                     * (reward + GAMMA * Q - self.lastQ) \
                                     * self.lastFeatures[feature]
            if self.weights[feature] < 0:
                self.weights[feature] = 0

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

    def nextIndex(self, index):
        index = index + 1 if index < 3 else 0
        while self.visibleAgents[index] is None:
            index = index + 1 if index < 3 else 0
        return index

    def getReward(self, gameState, preGameState):
        """
        Overwrite
        """
        return 0

    def getFeatures(self, gameState):
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


class OffensiveReflexAgent(MonteCarloAgent):

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

    def getFeatures(self, gameState):

        features = util.Counter()
        # gameState = getSuccessor(gameState, self.index, action)
        myPos = gameState.getAgentPosition(self.index)
        numCarryingFood = gameState.getAgentState(self.index).numCarrying
        foodList = self.getFood(gameState).asList()

        # feature: 'num of food'
        # feature: 'dist to closest food'
        if len(foodList) > 2:
            features['num of food'] = - len(foodList) / self.totalNumOppositeFood
            features['dist to closest food'] = - min(
                [self.getMazeDistance(myPos, food) for food in foodList]) / self.mapArea

        # feature: 'dist to closest ghost'
        distsToGhosts = [self.getMazeDistance(myPos, pos)
                         for index, pos in enumerate(self.enemiesPos)
                         if index in self.getOpponents(gameState)
                         and pos is not None
                         and not gameState.getAgentState(index).isPacman
                         and gameState.getAgentState(index).scaredTimer < 3
                         and self.getMazeDistance(myPos, pos) < 6]
        if len(distsToGhosts) > 0:
            features['dist to closest ghost'] = ((10 + numCarryingFood) / self.totalNumOppositeFood) \
                                                * min(distsToGhosts) / self.mapArea

        # feature 'dist to closest capsule'
        # feature 'num of capsules'
        flag = [True if gameState.getAgentState(index).scaredTimer < dist < 10 else False
                for index, dist in enumerate(gameState.getAgentDistances())
                if index in self.getOpponents(gameState)]
        if True in flag:
            distToCapsules = [self.getMazeDistance(myPos, pos) for pos in self.getCapsules(gameState)]
            if len(distToCapsules) > 0:
                features['dist to closest capsule'] = - min(distToCapsules) / self.mapArea
                features['num of capsules'] = - len(self.getCapsules(gameState)) / self.totalNumOppositeCapsules

        # feature 'dist to mid line'
        distToMidLine = [self.getMazeDistance(myPos, each) for each in self.midLine]
        features['dist to mid line'] = - (min(distToMidLine) / self.mapArea) \
                                       * (numCarryingFood / self.totalNumOppositeFood)

        # Normalize and return
        features.divideAll(len(self.weights))

        return features

    def getWeights(self):

        """
        Submission version
        """
        # sys.path.append('teams/Pacman_Go/')
        # self.file = os.path.join("./offensive.json")

        self.file = os.path.join(os.path.dirname(__file__), "offensive.json")

        f = open(self.file, "r")
        data = f.read().splitlines()[-1]
        weights = json.loads(data)
        f.close()

        return weights


class DefensiveReflexAgent(MonteCarloAgent):

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

        # enemy does nothing
        if capsuleChange == 0 and foodChange == 0:
            reward += 0.02

        # eat enemy
        myPos = gameState.getAgentPosition(self.index)
        enemies = [preGameState.getAgentState(i) for i in self.getOpponents(preGameState)]
        enemiesPrePos = [e.getPosition for e in enemies if e.getPosition() is not None]
        if myPos in enemiesPrePos:
            reward += 5

        return reward

    def getFeatures(self, gameState):

        features = util.Counter()
        # gameState = getSuccessor(gameState, self.index, action)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # feature 'dist to invader'
        distsToInvaders = [self.getMazeDistance(myPos, pos)
                           for index, pos in enumerate(self.enemiesPos)
                           if index in self.getOpponents(gameState)
                           and pos is not None
                           and gameState.getAgentState(index).isPacman]
        if myState.scaredTimer > 0:
            distsToInvaders = [abs(dist - 2) for dist in distsToInvaders]
        if len(distsToInvaders) > 0:
            features['dist to invader'] = - min(distsToInvaders) / self.mapArea

        # feature 'num of Invaders'
        invaders = [gameState.getAgentState(i).isPacman for i in self.getOpponents(gameState)]
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

        """
        Submission version
        """
        # sys.path.append('teams/Pacman_Go/')
        # self.file = os.path.join("./defensive.json")

        self.file = os.path.join(os.path.dirname(__file__), "defensive.json")

        f = open(self.file, "r")
        data = f.read().splitlines()[-1]
        weights = json.loads(data)
        f.close()

        return weights


class Tree:

    def __init__(self, index, state, action, depth, r, parent=None):
        self.parent = parent
        self.children = []
        self.index = index  # index that will make a move in this node
        self.state = state  # gameState
        self.action = action  # action from parent
        self.depth = depth
        self.V = -np.inf  # evaluation
        self.r = r  # rewared
        self.N = 0  # visited times

    # def getBestChild(self):
    #     maxV = -np.inf
    #     node = None
    # if node.index in self.getTeam(node.state):
    #     for child in self.children:
    #         if child.V > maxV:
    #             maxV = child.V
    #             node = child
    #     return node

def getSuccessor(gameState, index, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(index, action)
    pos = successor.getAgentPosition(index)
    if pos != nearestPoint(pos):
        return successor.generateSuccessor(index, action)
    else:
        return successor
