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
import random, time, util
from game import Directions
import game

import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GeneralAgent', second = 'DefensiveReflexAgent'):
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

maxDeepth = 15

def getDirection(start,nextStep):
    x1,y1 = start
    x2,y2 = nextStep
    if x1+1 == x2 and y1 == y2:
        return 'East'
    elif x1-1 == x2 and y1 == y2:
        return 'West'
    elif x1 == x2 and y1+1 == y2:
        return 'North'
    elif x1 == x2 and y1-1 == y2:
        return 'South'
    return 'Error'
def getPosByDirection(start,direction):
    x,y = start
    if direction == 'East':
        return (x+1,y)
    elif direction == 'West':
        return (x-1,y)
    elif direction == 'North':
        return (x,y+1)
    elif direction == 'South':
        return (x+1,y-1)
    return 'Error'

def getNearPoints(pos,gameState):
    x,y = pos
    print(x,y)
    temp = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    return [(x,y) for x,y in temp if not gameState.hasWall(x,y)]

#####################
# Structure for UCT #
#####################
class Node(object):

    def __init__(self,gameState,agent,action,parentNode,ghostPos,borderline):
        self.parentNode = parentNode
        self.action = action
        if parentNode == None:
            self.deepth = 0
        else:
            self.deepth = parentNode.deepth + 1

        self.child = []
        self.v_times = 1
        self.q_value = 0.0

        self.v_back = 1
        self.q_back = 0.0

        self.gameState = gameState.deepCopy()
        self.ghostPos = ghostPos
        self.legalActions = gameState.getLegalActions(agent.index)
        self.illegalActions = []
        self.legalActions.remove('Stop')

        self.legalActions = list(set(self.legalActions)-set(self.illegalActions))

        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline

        self.agent = agent
        self.E = 0.95

def getBestChild(node):
    bestScore = -99999
    bestChild = None
    for n in node.child:
        score = n.q_value/n.v_times
        if score > bestScore:
            bestScore = score
            bestChild = n
    return bestChild

def getExpandedNode(node):
    if node.deepth >= maxDeepth:
        return node

    if node.unexploredActions != []:
        action = node.unexploredActions.pop()
        tempGameState = node.gameState.deepCopy()
        nextGameState = tempGameState.generateSuccessor(node.agent.index,action)
        childNode = Node(nextGameState,node.agent,action,node,node.ghostPos,node.borderline)
        node.child.append(childNode)
        return childNode
    
    if util.flipCoin(node.E): # E-greedy 
        nextBestNode = getBestChild(node)
    else:
        nextBestNode = random.choice(node.child)
    return getExpandedNode(nextBestNode)

def getReward(node):
    # lastPos = node.parentNode.gameState.getAgentPosition(node.agent.index)
    nowPos = node.gameState.getAgentPosition(node.agent.index)
    if nowPos == node.gameState.getInitialAgentPosition(node.agent.index):
        return -500
    
    dis_to_ghost = min(node.agent.getMazeDistance(nowPos,ghost_pos) for ghost_pos in node.ghostPos)
    if dis_to_ghost <= node.deepth:
        return -500

    value = getFeaturesAttack(node.agent,node)*getWeight()
    return value
    
def backpropagation(node,reward):
    node.v_times += 1
    node.q_value += reward
    if node.parentNode != None:
        backpropagation(node.parentNode,reward)

def MCTS(node):
    timeLimit = 0.5
    start = time.time()
    while(time.time()-start < timeLimit):
    # for i in range(maxTreeIteration):
        
        nodeForSimulation = getExpandedNode(node) #selection and expand

        reward = getReward(nodeForSimulation)

        backpropagation(nodeForSimulation,reward)
    
    return getBestChild(node).action

def getFeaturesAttack(agent,node):
    """
    Returns a counter of features for the state
    """
    gameState = node.gameState
    lastGameState = node.parentNode.gameState
    features = util.Counter()

    features['getFood'] = gameState.getAgentState(agent.index).numCarrying \
                          - lastGameState.getAgentState(agent.index).numCarrying
    
    # if features['getFood'] == 0:
    #     if len(agent.getFood(gameState).asList()) > 0:
    features['minDistToFood'] = agent.getMinDistToFood(gameState)

    return features

def getWeight():
    return {'minDistToFood':-10,'getFood':100}
# --------------------------------------------------------------------------
def getExpandedNode_back(node):
    if node.deepth >= maxDeepth:
        return node

    if node.unexploredActions != []:
        action = node.unexploredActions.pop()
        tempGameState = node.gameState.deepCopy()
        nextGameState = tempGameState.generateSuccessor(node.agent.index,action)
        childNode = Node(nextGameState,node.agent,action,node,node.ghostPos,node.borderline)
        node.child.append(childNode)
        return childNode
    
    if util.flipCoin(node.E): # E-greedy 
        nextBestNode = getBestChild_back(node)
    else:
        nextBestNode = random.choice(node.child)
    return getExpandedNode_back(nextBestNode)

def MCTS_back(node):
    timeLimit = 0.9
    start = time.time()
    while(time.time()-start < timeLimit):
        
        nodeForSimulation = getExpandedNode_back(node) 

        reward = getReward_back(nodeForSimulation)

        backpropagation_back(nodeForSimulation,reward)
    
    return getBestChild_back(node).action

def getBestChild_back(node):
    bestScore = -99999
    bestChild = None
    for n in node.child:
        score = n.q_back/n.v_back
        if score > bestScore:
            bestScore = score
            bestChild = n
    return bestChild

def backpropagation_back(node,reward):
    node.v_back += 1
    node.q_back += reward
    if node.parentNode != None:
        backpropagation_back(node.parentNode,reward)

def getReward_back(node):
    nowPos = node.gameState.getAgentPosition(node.agent.index)
    if nowPos == node.gameState.getInitialAgentPosition(node.agent.index):
        return -1000
    value = getFeature_back(node.agent,node) * getWeight_back()
    return value

def getFeature_back(agent,node):
    gameState = node.gameState
    feature = util.Counter()
    nowPos = node.gameState.getAgentPosition(node.agent.index)
    feature['distance'] = min([node.agent.getMazeDistance(nowPos,borderPos) for borderPos in node.borderline])
    return feature

def getWeight_back():
    return {'distance':-1}
##########
# Agents #
##########

class GeneralAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

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

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.totalFoodNum = float(len(self.getFood(gameState).asList()))

        self.needFallback = False

        self.mapwidth = gameState.data.layout.width
        self.mapheight = gameState.data.layout.height
        self.myBorders = self.getMyBorder(gameState)
        self.enemyBorders = self.getEnemyBorder(gameState)
        # self.oneKi,self.twoKi,self.threeKi,self.fourKi = self.pointClassification(gameState)
        # self.dangerPath = self.getDangerPath(gameState)


    def getMyBorder(self,gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            borderLine = [(self.mapwidth/2-1,h) for h in range(self.mapheight)]
            return [(x,y) for (x,y) in borderLine if (x,y) not in walls and (x+1,y) not in walls]
        else:
            borderLine = [(self.mapwidth/2,h) for h in range(self.mapheight)]
            return [(x,y) for (x,y) in borderLine if (x,y) not in walls and (x-1,y) not in walls]

    def getEnemyBorder(self,gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            borderLine = [(self.mapwidth/2,h) for h in range(self.mapheight)]
            return [(x,y) for (x,y) in borderLine if (x,y) not in walls and (x-1,y) not in walls]
        else:
            borderLine = [(self.mapwidth/2-1,h) for h in range(self.mapheight)]
            return [(x,y) for (x,y) in borderLine if (x,y) not in walls and (x+1,y) not in walls]

    def getEnemyGhost(self,gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList
    
    def getDangerouseGhost(self,gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        dangerGhosts = []
        ghosts = self.getEnemyGhost(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for g in ghosts:
            distance = self.getMazeDistance(myPos,gameState.getAgentPosition(g))
            if distance <=5:
                dangerGhosts.append(g)
        return dangerGhosts

    def getEnemyPacman(self,gameState):
        """
        Return Observable Oppo-Pacman Position
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if enemyState.isPacman and gameState.getAgentPosition(enemy) != None:
                enemyList.append(enemy)
        return enemyList

    # def isAttackMode(self,gameState):
    #     absoluteScore = self.getScore(gameState)
    #     if absoluteScore <= self.totalFoodNum/2:
    #         attFoodNum = len(self.getFood(gameState).asList())
    #         if float(attFoodNum)/self.totalFoodNum >= 0.2:
    #             return True
    #     else:
    #         if gameState.getAgentState(self.index).isPacman:
    #             self.needFallback = True
    #             return True
    #         else:
    #             return False

    def chooseAction(self, gameState):
        """
        Picks best actions.
        """
        start = time.time()
        actions = gameState.getLegalActions(self.index)
        if gameState.getAgentState(self.index).isPacman: # Collecting as Pacman
            ghostPos = [gameState.getAgentPosition(g) for g in self.getDangerouseGhost(gameState)]
            if len(self.getFood(gameState).asList()) < 2:
                rootNode = Node(gameState,self,None,None,ghostPos,self.myBorders)
                resultAction = MCTS_back(rootNode)
            elif len(ghostPos) == 0:
                values = [self.evaluateAttack(gameState, a) for a in actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                resultAction = random.choice(bestActions)
                # rootNode = Node(gameState,self,None,None,ghostPos)
            elif gameState.getAgentState(self.index).numCarrying > 7:
                print(1111111111111)
                rootNode = Node(gameState,self,None,None,ghostPos,self.myBorders)
                resultAction = MCTS_back(rootNode)
            else:
                rootNode = Node(gameState,self,None,None,ghostPos,self.myBorders)
                print(2222222222222)
                resultAction = MCTS(rootNode)
        else: # On the way to attack as Ghosts
            ghosts = self.getEnemyGhost(gameState)
            values = [self.evaluate(gameState, a, ghosts) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            resultAction = random.choice(bestActions)

        return resultAction
    

    def evaluateAttack(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeaturesAttack(gameState, action)
        weights = self.getWeightsAttack(gameState, action)
        return features * weights

    def getFeaturesAttack(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successorGameState = self.getSuccessor(gameState, action)
        if successorGameState.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(successorGameState).asList()) > 0:
                features['minDistToFood'] = self.getMinDistToFood(successorGameState)
        return features

    def getWeightsAttack(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1,'getFood': 100}

    def evaluate(self, gameState, action, ghosts):
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
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)#self.getScore(successor)

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def getMinDistToFood(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(myPos,f) for f in self.getFood(gameState).asList()])


class DefensiveReflexAgent(GeneralAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
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

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
