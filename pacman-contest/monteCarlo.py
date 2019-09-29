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
               first='OffensiveReflexAgent', second='OffensiveReflexAgent'):
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

class ValueIteration:

    def __init__(self, gameState, index, epoch, discount=0.9):
        self.index = index
        self.isRed = gameState.isOnRedTeam(index)
        self.discount = discount
        self.width, self.height = gameState.getWalls().width, gameState.getWalls().height
        self.rewards = np.zeros((self.width, self.height), dtype=None)
        self.Vs = np.zeros((self.width, self.height), dtype=None)
        self.policies = np.full((self.width, self.height), None)
        self.toUpdate = []

        self.buildVMap(gameState)
        self.iteration(epoch)
        self.buildPoliciesMap()

    def buildVMap(self, gameState):
        walls = gameState.getWalls().asList()
        foods = gameState.getBlueFood().asList() if self.isRed else gameState.getRedFood().asList()
        capsules = gameState.getBlueCapsules() if self.isRed else gameState.getRedCapsules().asList()
        width, height = gameState.getWalls().width, gameState.getWalls().height

        # build reward map
        for x in range(width):            # set out boundary cell to None
            self.rewards[x][0] = None
            self.rewards[x][-1] = None
        for y in range(height):           # set out boundary cell to None
            self.rewards[0][y] = None
            self.rewards[-1][y] = None
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

class MonteCarloAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.gamma = 0.8

        CaptureAgent.registerInitialState(self, gameState)
        self.myTeamIndexies= self.getTeam(gameState)
        # print("register success: index ", self.index)

    def chooseAction(self, gameState):
        return self.monteCarloSearch(gameState)

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
    def monteCarloSearch(self, gameState):
        print("\n=================================NEW TURN====================================")

        valueIter = ValueIteration(gameState, self.index, 100)

        self.toExpand = []
        self.root = Tree(i=self.index, s=gameState, a=None, mapRewards=valueIter.rewards, mapPolicies=valueIter.policies)
        if gameState.getLegalActions(self.index) != 0:  # for case if born in an island
            self.toExpand.append(self.root)

        # print(self.root.s.getAgentPosition(self.root.i), self.root.i, self.root.d)
        start = time.time()
        n = 0

        while True:
            n = n+1

            # print("\n==========select=============")
            selectNode, selectAction = self.selectNode()
            # print("==========expandNode=========")
            node = self.expandNode(selectNode, selectAction)
            # print("==========simulate===========")
            V = self.simulate(node)
            # print("==========backprop===========")
            self.backprop(V, node)
            # print("\n")
            end = time.time()
            if end - start > 0.9 :
                break
        maxV = self.root.children[0].V
        decision = self.root.children[0].a
        for child in self.root.children:
            if child.V > maxV:
                maxV = child.V
                decision = child.a
        """=================debug=================="""
        for child in self.root.children:
            if child.a == Directions.NORTH:
                print("NORTH:", self.childCount(child), child.V)
            elif child.a == Directions.SOUTH:
                print("SOUTH:", self.childCount(child), child.V)
            elif child.a == Directions.WEST:
                print("WEST:", self.childCount(child), child.V)
            elif child.a == Directions.EAST:
                print("EAST:", self.childCount(child), child.V)
            elif child.a == Directions.STOP:
                print("STOP:", self.childCount(child), child.V)

        print("total: ", n, self.root.s.getAgentPosition(self.root.i), self.root.i, decision)
        return decision

    def childCount(self, root):
        count = len(root.children)
        for each in root.children:
            count += self.childCount(each)
        return count

    def selectNode(self):
        while len(self.toExpand) > 0:  # for case if have developed all of states
            node = random.choice(self.toExpand)
            actions = node.s.getLegalActions(node.i)
            if len(actions) == 0:  # jump to begin if this node is dead end
                self.toExpand.remove(node)
                continue

            # remove actions if that leads to an existed child
            toRemove = []
            for a in actions:
                for child in node.children:
                    if child.a == a:
                        toRemove.append(a)
            for each in toRemove:
                actions.remove(each)

            if len(actions) <= 0:
                self.toExpand.remove(node)
                continue
            elif len(actions) == 1:
                self.toExpand.remove(node)

            action = random.choice(actions)
            # print(node.s.getAgentPosition(node.i), node.i, actions, "=>", action)
            return node, action

    def expandNode(self, parent, action):
        # print(parent.s.getAgentPosition(parent.i), parent.i)
        successorState = self.getSuccessor(parent.s, parent.i, action)
        """ evaluate reward """
        (x, y) = successorState.getAgentPosition(parent.i)
        reward = parent.mapRewards[x, y]
        nextIndex = self.myTeamIndexies[0] if parent.i == self.myTeamIndexies[1] else self.myTeamIndexies[1]
        # if any reward on the map is changed, make up a new rewards and policies map
        if reward == 0:
            successor = Tree(i=nextIndex, s=successorState, a=action, parent=parent, r=reward, mapRewards=parent.mapRewards, mapPolicies=parent.mapPolicies)
        else:
            valueIter = ValueIteration(parent, self.index, 100)
            successor = Tree(i=nextIndex, s=successorState, a=action, parent=parent, r=reward, mapRewards=valueIter.rewards, mapPolicies=valueIter.policies)
        parent.children.append(successor)
        self.toExpand.append(successor)
        # print("expand Node from", parent.s.getAgentPosition(parent.i), action, "to ", successor.s.getAgentPosition(parent.i), "index: ", parent.i)
        return successor

    def simulate(self, node):
        index = node.i
        n = 0
        reward = 0
        successor = node.s
        mapRewards = node.mapRewards
        mapPolicies = node.mapPolicies
        # mapRewards, mapPolicies = [], []
        # for i in range(4):
        #     if i in self.myTeamIndexies:
        #         valueIter = ValueIteration(successor, i, 100)
        #         mapRewards.append(valueIter.rewards)
        #         mapPolicies.append(valueIter.policies)
        #     else:
        #         mapRewards.append(None)
        #         mapPolicies.append(None)

        (x, y) = successor.getAgentPosition(index)

        while not successor.isOver():  # game not terminate yet
            # print(mapPolicies)
            # print(mapRewards)
            # print(successor.getAgentPosition(index), successor.getLegalActions(index))
            action = mapPolicies[x, y]
            n = n + 1
            successor = self.getSuccessor(successor, index, action)
            index = self.myTeamIndexies[0] if index == self.myTeamIndexies[1] else self.myTeamIndexies[1]  # next index
            (x, y) = successor.getAgentPosition(index)
            reward = mapRewards[x, y]
            if reward > 0.01:
                if index == self.index:
                    break
                # else:
                #     mapRewards, mapPolicies = [], []
                #     for i in range(4):
                #         if i in self.myTeamIndexies:
                #             valueIter = ValueIteration(successor, i, 50)
                #             mapRewards.append(valueIter.rewards)
                #             mapPolicies.append(valueIter.policies)
                #         else:
                #             mapRewards.append(None)
                #             mapPolicies.append(None)


        # print(node.s.getAgentPosition(index), index)
        # print(node.r + (self.gamma ** n) * reward)
        # print(n)
        # print(self.gamma ** n)
        # print(self.gamma ** n * reward)
        return node.r + (self.gamma ** n) * reward  # r(s) + gamma**n * r(s_n)

    def backprop(self, V, successor):
        successor.V = V  # reward + discounted future reward
        successor.N += 1

        node = successor.parent
        while node is not None:
            reward = None
            maxV = max([child.V for child in node.children])
            for child in node.children:  # max(V(s'))
                if child.V == maxV:
                    reward = child.r
            node.V = reward + self.gamma * maxV
            node.N += 1
            node = node.parent



    def getSuccessor(self, gameState, index, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        # print("generating: ", gameState.getAgentPosition(index), index, action)
        successor = gameState.generateSuccessor(index, action)
        pos = successor.getAgentPosition(index)
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            # # print("pos != nearestPoint(pos)")
            return successor.generateSuccessor(index, action)
        else:
            return successor

    def evaluate(self, gameState, index, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, index, action)
        weights = self.getWeights(gameState, index, action)
        return features * weights

    def getFeatures(self, gameState, index, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        features['score'] = self.getScore(gameState)
        return features

    def getWeights(self, gameState, index, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'score': 1.0}


class OffensiveReflexAgent(MonteCarloAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, index, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, index, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, index, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class Tree:
    def __init__(self, i, s, a, mapRewards, mapPolicies, r=0, parent=None):
        self.parent = parent
        self.children = []
        self.i = i  # index that takes move in this node
        self.s = s  # gameState
        self.a = a  # action from parent
        self.V = 0  # evaluation
        self.N = 0  # visted
        self.r = r  # reward
        self.mapRewards = mapRewards
        self.mapPolicies = mapPolicies
