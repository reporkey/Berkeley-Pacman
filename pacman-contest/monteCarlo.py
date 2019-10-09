# baselineTeam.py

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

class MonteCarloAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.gamma = 1
        self.Cp = 0.7       # UCT
        self.epsilon = 0.1  # e-greedy
        self.init()
        CaptureAgent.registerInitialState(self, gameState)

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
        # print("=================================NEW TURN====================================")

        self.toExpand = []
        self.root = Tree(i=self.index, s=gameState, a=None)
        if len(gameState.getLegalActions(self.index)) != 0:  # for case if born in an island
            self.toExpand.append(self.root)

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
            if end - start > 0.5:
                break
        maxIndex = np.argmax([self.gamma * child.V for child in self.root.children])
        decision = self.root.children[int(maxIndex)].a
        """=================debug=================="""
        if self.role == "Offensive":
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

            print("total: ", n, self.root.s.getAgentPosition(self.root.i), self.root.i, decision, "\n")
        return decision

    def selectNode(self):
        node = self.root
        action = None
        actions = node.s.getLegalActions(node.i)

        while (len(node.children) > 0 or len(actions) > 0) and not node.s.isOver():

            # choose action
            existedActions = set([each.a for each in node.children])
            actions = list(set(actions) - existedActions)
            if len(actions) > 0:
                action = np.random.choice(actions)
                break

            """ UCB1 (UCT) """
            """ Since the Q is not normalized between attacker and defender, UCT is not a good choice """
            # UCB1 = np.array([child.V + 2 * self.Cp * np.sqrt(2 * np.log(node.N) / child.N) for child in node.children])
            # node = node.children[np.random.choice(np.flatnonzero(UCB1 == UCB1.max()))]
            # actions = node.s.getLegalActions(node.i)

            """ e-greedy """
            if np.random.random() < self.epsilon:
                node = np.random.choice(node.children)
            else:
                Qvalues = np.array([child.V for child in node.children])
                node = node.children[np.random.choice(np.flatnonzero(Qvalues == Qvalues.max()))]
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
        successorState = getSuccessor(parent.s, parent.i, action)
        nextIndex = parent.i + 1 if parent.i < 3 else 0
        while successorState.getAgentPosition(nextIndex) is None:
            nextIndex = nextIndex + 1 if nextIndex < 3 else 0
        successor = Tree(i=nextIndex, s=successorState, a=action, parent=parent)
        parent.children.append(successor)
        self.toExpand.append(successor)
        # print("expand Node from", parent.s.getAgentPosition(parent.i), action, "to ", successor.s.getAgentPosition(parent.i), "index: ", parent.i)
        return successor

    def simulate(self, node):
        if node.parent.i == self.index:
            return self.evaluate(node.parent.s, node.a)
        else:
            return -np.inf

    def backprop(self, V, successor):
        successor.V = V  # discounted future reward
        successor.N += 1

        node = successor.parent
        while node is not None:
            node.V = np.max([node.V] + [(self.gamma * child.V) for child in node.children]) # max among (self v & childrens' V)
            node.N += 1
            node = node.parent

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
        successor = getSuccessor(gameState, self.index, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def init(self):
        self.role = ""


class OffensiveReflexAgent(MonteCarloAgent):

    def init(self):
        self.role = "Offensive"

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = getSuccessor(gameState, self.index, action)
        foodList = self.getFood(successor).asList()
        features['#Food'] = len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'#Food': -100, 'distanceToFood': -1}


class DefensiveReflexAgent(MonteCarloAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def init(self):
        self.role = "Defensive"

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = getSuccessor(gameState, self.index, action)

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


class Tree:
    def __init__(self, i, s, a, parent=None):
        self.parent = parent
        self.children = []
        self.i = i  # index that will make a move in this node
        self.s = s  # gameState
        self.a = a  # action from parent
        self.V = -np.inf  # evaluation
        self.N = 0  # visited times
        # self.r = r  # reward


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
