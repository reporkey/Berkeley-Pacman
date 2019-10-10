
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

from game import *
from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

class OffensiveReflexAgent(CaptureAgent):


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
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    # self.epsilon = 0.0  # exploration prob
    self.epsilon = 0.05
    # self.alpha = 0.2  # learning rate
    self.alpha = 0.2
    self.discountRate = 0.8
    # self.weights = {'closest-food': -2.2558226236802597,
                   # 'bias': 1.0856704846852672,
                   # '#-of-ghosts-1-step-away': -0.18419418670562,
                   # 'successorScore': -0.027287497346388308,
                   # 'eats-food': 9.970429654829946}
    self.weights = {'closest-food': -2.2558226236802597,
                    'bias': 1.0856704846852672,
                    '#-of-ghosts-1-step-away': -18.419418670562,
                    'successorScore': -0.027287497346388308,
                    'eats-food': 9.970429654829946}

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

  def QValue(self, gameState, action):
    features = self.getFeatures(gameState, action)
    return features * self.weights

  def getQValue(self, gameState):
    Q_Value = []
    all_Possible_Action = gameState.getLegalActions(self.index)

    if not len(all_Possible_Action) == 0:

      for action in all_Possible_Action:
        Q_Value.append(self.QValue(gameState, action))
      return max(Q_Value)

  def update(self, gameState, action, nextState, reward):
    """
    features = self.getFeatures(gameState, move)
    features_list = features.sortedKeys()
    counter = 0
    for feature in features:
      difference = 0
      if len(gameState.getLegalActions(nextState)) == 0:
        difference = reward - self.getQValue(gameState, move)
      else:
        difference = (reward + self.discountRate * max([self.getQValue(nextState, next_move)
                                                        for next_move in gameState.getLegalActions(nextState)])) \
                     - self.getQValue(gameState, move)
      self.weights[feature] = self.weights[feature] + self.alpha * difference * difference * features[feature]
      counter += 1
    """


    """
    nextStateValue = self.getQValue(nextState)
    stateValue = self.QValue[(gameState, action)]
    learnRate = self.alpha
    discountRate = self.discountRate
    learnedValue = reward + discountRate * nextStateValue
    self.QValue[(gameState, action)] = stateValue + (learnRate * (learnedValue - stateValue)) 
    """

    first_part = (1 - self.alpha) * self.getQValue(state, action)
    if len(self.getLegalActions(nextState)) == 0:
      sample = reward
    else:
      sample = reward + (self.discountRate * max(
        [self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
    second_part = self.alpha * sample
    self.QValue[(state, action)] = first_part + second_part


  def getPolicy(self, gameState):
    values = []
    all_Possible_Action = gameState.getLegalActions(self.index)
    all_Possible_Action.remove(Directions.STOP)
    if len(all_Possible_Action) == 0:
      return None
    else:
      for action in all_Possible_Action:
        # self.updateWeights(gameState, action)
        Q_Val = self.QValue(gameState, action)
        values.append((Q_Val, action))
    return max(values)[1]

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

  def chooseAction(self, gameState):
    # Pick Action
    legalActions = gameState.getLegalActions(self.index)
    action = None

    if len(legalActions) != 0:
      """
      prob = util.flipCoin(self.epsilon)
      if prob:
        action = random.choice(legalActions)
      else:
      """
      action = self.getPolicy(gameState)
    return action

  def getFeatures(self, gameState, action):
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

  def getWeights(self, gameState, action):
    """
    return {'successorScore': 100, 'distanceToFood': -1}
    """

    return self.weights

  def updateWeights(self, gameState, action):
    features = self.getFeatures(gameState, action)
    nextState = self.getSuccessor(gameState, action)

    # Calculate the reward. NEEDS WORK
    reward = nextState.getScore() - gameState.getScore()

    for feature in features:
      correction = (reward + self.discountRate * self.getValue(nextState)) - self.getQValue(gameState, action)
      self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

  def closestFood(self, pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class DefensiveReflexAgent(OffensiveReflexAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    """
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    """
    CaptureAgent.registerInitialState(self, gameState)
    self.myAgents = CaptureAgent.getTeam(self, gameState)
    self.opAgents = CaptureAgent.getOpponents(self, gameState)
    self.myFoods = CaptureAgent.getFood(self, gameState).asList()
    self.opFoods = CaptureAgent.getFoodYouAreDefending(self, gameState).asList()

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features
    """
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
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    """
    return {'successorScore': 1.0}
    """
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



