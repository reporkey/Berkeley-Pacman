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

    def monteCarloSearch(self, gameState):
        # print("\n=================================NEW TURN====================================")
        self.toExpand = []
        self.root = Tree(i=self.index, s=gameState, a=None)
        if gameState.getLegalActions(self.index) != 0:  # for case if born in an island
            self.toExpand.append(self.root)

        # print(self.root.s.getAgentPosition(self.root.i), self.root.i, self.root.d)
        start = time.time()
        n = 0

        while True:
            n = n+1

            # print("==========select=============")
            selectNode, selectAction = self.selectNode()
            # print("==========expandNode=========")
            node = self.expandNode(selectNode, selectAction)
            # print("==========simulate===========")
            V = self.simulate(node)
            # print("==========backprop===========")
            self.backprop(V, node)
            # print("\n")
            end = time.time()
            if end - start > 0.8:
                break
        decision = self.root.d
        print("n: ", n, self.root.s.getAgentPosition(self.root.i), self.root.i, self.root.d)
        # print("==========crop leaf==========")
        # for child in self.root.children:
        #     if child.a is self.root.d:
        #         self.root = child
        # print("new location:", self.root.s.getAgentPosition(self.root.i))
        return decision

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
        reward = self.evaluate(parent.s, parent.i, action)
        successorState = self.getSuccessor(parent.s, parent.i, action)
        i = self.myTeamIndexies[0] if parent.i == self.myTeamIndexies[1] else self.myTeamIndexies[1]
        successor = Tree(i=i, s=successorState, a=action, parent=parent, r=reward)
        parent.children.append(successor)
        self.toExpand.append(successor)
        # print("expand Node from", parent.s.getAgentPosition(self.index), action, "to ", successor.s.getAgentPosition(self.index), "index: ", self.index)
        return successor

    def simulate(self, node):
        index = node.i
        n = 0
        reward = 0
        successor = node.s

        actions = successor.getLegalActions(index)
        while len(actions) > 1 and not successor.isOver() and reward == 0:  # any step except stop, & game not terminate yet
            action = random.choice(actions)  # randomly simulate
            n = n + 1
            if index == node.i:     # evaluate if only this action is done by myself (the agent is asked to move)
                reward = self.evaluate(successor, index, action)
            successor = self.getSuccessor(successor, index, action)
            index = self.myTeamIndexies[0] if index == self.myTeamIndexies[1] else self.myTeamIndexies[1]
            actions = successor.getLegalActions(index)
        return node.r + self.gamma ** n * reward  # r(s) + gamma**n * r(s_n)

    def backprop(self, V, successor):
        successor.V = V  # reward + discounted future reward
        successor.N += 1

        node = successor.parent
        while node is not None:
            maxV = node.children[0].V
            decision = node.children[0].a  # best decision
            reward = node.children[0].r
            for child in node.children:  # max(V(s'))
                # if node is self.root:
                #     print(child.V, child.a, child.r)
                if child.V > maxV:
                    maxV = child.V
                    decision = child.a
                    reward = child.r
            node.V = reward + self.gamma * maxV
            node.d = decision
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
    def __init__(self, i, s, a, r=0, d=None, parent=None):
        self.parent = parent
        self.children = []
        self.i = i  # index
        self.s = s  # gameState
        self.a = a  # action from parent
        self.d = d  # decision action
        self.V = 0  # evaluation
        self.N = 0  # visted
        self.r = r  # reward
