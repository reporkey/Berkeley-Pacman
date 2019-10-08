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

class ValueIteration:

    def __init__(self, gameState, index, epoch, heuristic, discount):
        self.index = index
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
        capsules = gameState.getBlueCapsules() if self.isRed else gameState.getRedCapsules().asList()
        numCarrying = gameState.getAgentState(self.index).numCarrying
        deliveryLine = [(self.width // 2 - 1 if self.isRed else self.width // 2, y) for y in range(1, self.height)]

        # build reward map
        for x in range(self.width):  # set out boundary cell to None
            self.rewards[x][0] = None
            self.rewards[x][-1] = None
        for y in range(self.height):  # set out boundary cell to None
            self.rewards[0][y] = None
            self.rewards[-1][y] = None

        # evaluate heuristically
        for (x, y) in foods:  # set reward of each food as 10
            self.rewards[x][y] += heuristic["food"]
        for (x, y) in capsules:  # set reward of each capsule as 100
            self.rewards[x][y] += heuristic["capsule"]
        for (x, y) in deliveryLine:
            self.rewards[x][y] += heuristic["delivery"] * numCarrying
        for (x, y) in walls:
            self.rewards[x][y] = None

        # lable visible enemies
        enemyIndices = gameState.getBlueTeamIndices() if self.isRed else gameState.getBlueTeamIndices()
        for enemyIndex in enemyIndices:
            enemyState = gameState.getAgentState(enemyIndex)
            if enemyState.configuration is not None:
                x, y = enemyState.getPosition()
                # I'm Ghost, enemy is pacman
                if enemyState.isPacman and not gameState.getAgentState(self.index).isPacman:
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
                        self.rewards[int(x)][int(y)] += (heuristic["enemyGhost"] + heuristic["foodLostPenalty"] * numCarrying)

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


class MonteCarloAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.gamma = 0.8
        self.Cp = 0.7
        self.epoch = 100

        CaptureAgent.registerInitialState(self, gameState)
        self.myTeamIndexies = self.getTeam(gameState)
        # print("register success: index ", self.index)

    def chooseAction(self, gameState):
        # x, y = gameState.getAgentPosition(self.index)
        # vi = ValueIteration(gameState, self.index, self.epoch, self.getHeuristic(), self.gamma)
        # print(x,y, vi.policies[x][y])
        # return vi.policies[x][y]
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

        self.heuristic = self.getHeuristic()
        self.allyHeuristic = self.getAllyHeuristic()
        valueIter = [None, None, None, None]
        for i in range(4):
            if i in self.myTeamIndexies:
                if i == self.index:
                    valueIter[i] = ValueIteration(gameState, i, self.epoch, self.heuristic, self.gamma)
                else:
                    valueIter[i] = ValueIteration(gameState, i, self.epoch, self.allyHeuristic, self.gamma)
        self.toExpand = []
        self.root = Tree(i=self.index, s=gameState, a=None, VI=valueIter)
        if gameState.getLegalActions(self.index) != 0:  # for case if born in an island
            self.toExpand.append(self.root)

        # print(self.root.s.getAgentPosition(self.root.i), self.root.i, self.root.d)
        start = time.time()
        n = 0

        while True:
            n = n + 1

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
            if end - start > 1:
                break
        maxIndex = np.argmax([child.r+self.gamma*child.V for child in self.root.children])
        decision = self.root.children[int(maxIndex)].a
        """=================debug=================="""
        for child in self.root.children:
            if child.a == Directions.NORTH:
                print("NORTH:\t", child.N, "\t", child.V)
            elif child.a == Directions.SOUTH:
                print("SOUTH:\t", child.N, "\t", child.V)
            elif child.a == Directions.WEST:
                print("WEST:\t", child.N, "\t", child.V)
            elif child.a == Directions.EAST:
                print("EAST:\t", child.N, "\t", child.V)
            elif child.a == Directions.STOP:
                print("STOP:\t", child.N, "\t", child.V)

        print("total: ", n, self.root.s.getAgentPosition(self.root.i), self.root.i, decision)
        # if decision == Directions.STOP:
        #     print(self.)
        return decision

    def selectNode(self):

        node = self.root
        action = None
        actions = node.s.getLegalActions(node.i)

        while len(node.children) > 0 or len(actions) > 0:

            # choose action
            existedActions = set([each.a for each in node.children])
            actions = list(set(actions) - existedActions)
            if len(actions) > 0:
                action = np.random.choice(actions)
                break

            """# Multi-armed bandit, UCB"""
            UCT = np.array([child.V + 2 * self.Cp * np.sqrt(2 * np.log(node.N) / child.N) for child in node.children])
            # print(UCT)
            # index = np.flatnonzero(UCT == UCT.max())
            # print([i for i in range(len(node.children))], "\t", index)
            node = node.children[np.random.choice(np.flatnonzero(UCT == UCT.max()))]
            # node = node.children[np.random.choice(range(len(node.children)))]
            actions = node.s.getLegalActions(node.i)

        # print(node.s.getAgentPosition(node.i), node.i, actions, "=>", action)
        return node, action

    # def selectNode(self):
    #     while len(self.toExpand) > 0:  # for case if have developed all of states
    #
    #         # TODO: Multi-armed bandit, UCB
    #
    #         node = random.choice(self.toExpand)
    #         actions = node.s.getLegalActions(node.i)
    #         if len(actions) == 0:  # jump to begin if this node is dead end
    #             self.toExpand.remove(node)
    #             continue
    #
    #         # remove actions if that leads to an existed child
    #         toRemove = []
    #         for a in actions:
    #             for child in node.children:
    #                 if child.a == a:
    #                     toRemove.append(a)
    #         for each in toRemove:
    #             actions.remove(each)
    #
    #         if len(actions) <= 0:
    #             self.toExpand.remove(node)
    #             continue
    #         elif len(actions) == 1:
    #             self.toExpand.remove(node)
    #
    #         action = random.choice(actions)
    #         # print(node.s.getAgentPosition(node.i), node.i, actions, "=>", action)
    #         return node, action

    def expandNode(self, parent, action):
        # print(parent.s.getAgentPosition(parent.i), parent.i)
        successorState = self.getSuccessor(parent.s, parent.i, action)
        """ evaluate reward """
        (x, y) = successorState.getAgentPosition(parent.i)
        reward = parent.VI[parent.i].rewards[x, y]
        # nextIndex = self.myTeamIndexies[0] if parent.i == self.myTeamIndexies[1] else self.myTeamIndexies[1]
        nextIndex = self.index
        # if any reward on the map is changed, make up a new rewards and policies map
        valueIter = [None, None, None, None]
        if reward == 0:
            valueIter = parent.VI
        else:
            for i in range(4):
                if i in self.myTeamIndexies:
                    if i == self.index:
                        valueIter[i] = ValueIteration(successorState, i, self.epoch, self.heuristic, self.gamma)
                    else:
                        valueIter[i] = ValueIteration(successorState, i, self.epoch, self.allyHeuristic, self.gamma)
        successor = Tree(i=nextIndex, s=successorState, a=action, parent=parent, r=reward, VI=valueIter)
        parent.children.append(successor)
        self.toExpand.append(successor)
        # print("expand Node from", parent.s.getAgentPosition(parent.i), action, "to ", successor.s.getAgentPosition(parent.i), "index: ", parent.i)
        return successor

    def simulate(self, node):

        # not simulate non-self
        # if node.i != self.index:
        #     return 0

        index = node.i
        # n = 0
        # reward = None
        successor = node.s
        mapRewards = node.VI[index].rewards
        mapPolicies = node.VI[index].policies
        (x, y) = successor.getAgentPosition(index)


        # while not successor.isOver() and n < 500:  # game not terminate yet
        #     action = mapPolicies[x, y]
        #     n = n + 1
        #     successor = self.getSuccessor(successor, index, action)
        #     (x, y) = successor.getAgentPosition(index)
        #     reward = mapRewards[x, y]
        #     if reward != 0:
        #         break

        # TODO: shaped reward, give penalty on "stop"
        # return (self.gamma ** n) * reward  # V(s) = gamma**n * r(s_n)
        return node.VI[index].Vs[x][y]  # V(s) = gamma**n * r(s_n)

    def backprop(self, V, successor):
        successor.V = V  # discounted future reward
        successor.N += 1

        node = successor.parent
        while node is not None:
            if node.i == self.index:
                # self: normal back up
                node.V = max([(child.r + self.gamma * child.V) for child in node.children])
            else:
                # Ally: add up with lower node
                node.V += max([(child.r+self.gamma*child.V) for child in node.children])
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

    def getAllyHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 200
        features['delivery'] = 20
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -1000
        features['enemyPacman'] = 200
        if features == self.heuristic:
            features['food'] = 100
            features['capsule'] = 50
            features['delivery'] = 20
            features['foodLostPenalty'] = -100
            features['enemyGhost'] = -1000
            features['enemyPacman'] = 5000
        return features


class OffensiveReflexAgent(MonteCarloAgent):

    def getHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 200
        features['delivery'] = 20
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -1000
        features['enemyPacman'] = 200
        return features


class DefensiveReflexAgent(MonteCarloAgent):

    def getHeuristic(self):
        features = util.Counter()
        features['food'] = 100
        features['capsule'] = 0
        features['delivery'] = 20
        features['foodLostPenalty'] = -100
        features['enemyGhost'] = -1000
        features['enemyPacman'] = 5000
        return features


class Tree:
    def __init__(self, i, s, a, VI, r=0, parent=None):
        self.parent = parent
        self.children = []
        self.i = i      # index that takes move in this node
        self.s = s      # gameState
        self.a = a      # action from parent
        self.V = 0      # evaluation
        self.N = 0      # visited times
        self.r = r      # reward
        self.VI = VI    # list of value iterations
