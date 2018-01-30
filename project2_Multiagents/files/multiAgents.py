# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def __init__(self):
    self.visitedPoints = []

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    position = gameState.getPacmanPosition()

    if legalMoves[chosenIndex] == 'West':
      self.visitedPoints.append((position[0]-1, position[1]))
    elif legalMoves[chosenIndex] == 'East':
      self.visitedPoints.append((position[0]+1, position[1]))
    elif legalMoves[chosenIndex] == 'North':
      self.visitedPoints.append((position[0], position[1]+1))
    elif legalMoves[chosenIndex] == 'South':
      self.visitedPoints.append((position[0], position[1]-1))
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    "for any unscared ghost, sum of their distance to pacman's position"
    
    
    evaluationValue = 0
    eval_ghostDistance = 0
    
    # for index in range(len(newScaredTimes)):
    #   distance = util.manhattanDistance(newPos, newGhostStates[index].getPosition())
    #   if newScaredTimes[index] > distance:
    #     eval_ghostDistance = eval_ghostDistance + (4/(distance+1)) 
    #   else:
    #     eval_ghostDistance = eval_ghostDistance - (4/(distance+1))

    # evaluationValue = evaluationValue + eval_ghostDistance

    if newPos in self.visitedPoints:
      evaluationValue = evaluationValue - self.visitedPoints.count(newPos)

    if action == 'Stop':
      evaluationValue = evaluationValue - 1
    
    for index in range(len(newScaredTimes)):
      distance = util.manhattanDistance(newPos, newGhostStates[index].getPosition())
      if newScaredTimes[index] > 0:
        if newScaredTimes[index] >= distance:
          if distance == 1:
            evaluationValue = evaluationValue + 20
          elif distance == 0:
            evaluationValue = evaluationValue + 40
      else:
        if distance == 1:
          evaluationValue = evaluationValue - 30
        elif distance == 0:
          evaluationValue = evaluationValue - 50



    if currentGameState.hasFood(newPos[0], newPos[1]):
      evaluationValue = evaluationValue + 20
      if successorGameState.isWin():
        evaluationValue = evaluationValue + 100
    
    nextLegalMoves = successorGameState.getLegalActions()
    # print nextLegalMoves
    for action in nextLegalMoves:
      # print action
      nextState = successorGameState.generatePacmanSuccessor(action)
      newPos = nextState.getPacmanPosition()
      if newFood[newPos[0]][newPos[1]]:
        evaluationValue = evaluationValue + 2
    # print evaluationValue

    # foodPositions = newFood.asList()
    # foodPositionsMean = [sum(x) / len(x) for x in zip(*foodPositions)]
    # if evaluationValue < 1:
    #   evaluationValue = evaluationValue + (10 / (util.manhattanDistance(newPos, foodPositionsMean)+1))
    return evaluationValue

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    "First-version"
    def maxValue(gameState, agentIndex, depth, ghostNum):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      value = float('-inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          value = max(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum))
      return value

    def minValue(gameState, agentIndex, depth, ghostNum):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      value = float('inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          if agentIndex == ghostNum:
            value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), 0, depth-1, ghostNum))
          else:
            value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum))
      return value

    bestValue = float('-inf')
    value = float('-inf')
    for action in gameState.getLegalActions(0):
      if action != 'Stop':
        value = max(value, minValue(gameState.generateSuccessor(0, action), 1, self.depth, gameState.getNumAgents()-1))
        if value > bestValue:
          bestAction = action
          bestValue = value
    return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    "========================================================="
    "Textbook-version"
    "========================================================="
    def maxValue(gameState, agentIndex, depth, ghostNum, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      value = float('-inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          value = max(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum, alpha, beta))
          if value >= beta:
            return value
          alpha = max(alpha, value)
      return value

    def minValue(gameState, agentIndex, depth, ghostNum, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      value = float('inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          if agentIndex == ghostNum:
            value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), 0, depth-1, ghostNum, alpha, beta))
          else:
            value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum, alpha, beta))
          if value <= alpha:
            return value
          beta = min(beta, value)
      return value

    
    alpha = float('-inf')
    beta = float('inf')
    ghostNum = gameState.getNumAgents()-1
    value = float('-inf')
    bestValue = float('-inf')
    bestAction = Directions.STOP

    for action in gameState.getLegalActions(0):
      if action != 'Stop':
        value = max(value, minValue(gameState.generateSuccessor(0, action), 1, self.depth, ghostNum, alpha, beta))
        if value >= beta:
          return action
        alpha = max(alpha, value)
        if value > bestValue:
          bestValue = value
          bestAction = action

    return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  visitedPoints = []

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    "*** YOUR CODE HERE ***"
    def maxValue(gameState, agentIndex, depth, ghostNum):
      value = float('-inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          value = max(value, expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum) - self.visitedPoints.count(gameState.getPacmanPosition()))
      return value

    def expValue(gameState, agentIndex, depth, ghostNum):
      value = 0
      legalActions = gameState.getLegalActions(agentIndex)
      for action in legalActions:
        if action != 'Stop':
          if agentIndex == ghostNum:
            value = value + expectimax(gameState.generateSuccessor(agentIndex, action), 0, depth-1, ghostNum)/float(len(legalActions))
          else: #agentIndex < ghostNum
            value = value + expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum)/float(len(legalActions))
      return value

    def expectimax(gameState, agentIndex, depth, ghostNum):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState) - self.visitedPoints.count(gameState.getPacmanPosition())
      if agentIndex == 0:
        return maxValue(gameState, agentIndex, depth, ghostNum)
      if agentIndex > 0:
        return expValue(gameState, agentIndex, depth, ghostNum)
    
    bestValue = float('-inf')
    bestAction = Directions.STOP
    for action in gameState.getLegalActions(0):
      if action != 'Stop':
        value = expectimax(gameState.generateSuccessor(0, action), 1, self.depth, gameState.getNumAgents()-1)
        # print 'value: ', value
        if value > bestValue:
          bestValue = value
          bestAction = action
    # print bestAction, ": ", bestValue
    # print "====================================="
    self.visitedPoints.append(gameState.getPacmanPosition())
    return bestAction


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  if currentGameState.isLose():
    return float('-inf')
  elif currentGameState.isWin():
    return float('inf')

  eval = [0, 0, 0, 0, 0]
  weight = [1, 1, 1, 1, 1]

  "=====Some useful information of currentGameState====="
  pacmanPosition = currentGameState.getPacmanPosition()
  pacmanLegalActions = currentGameState.getLegalActions(0)
  capsulesList = currentGameState.getCapsules()
  foods = currentGameState.getFood()
  foodList = foods.asList()
  ghostStates = currentGameState.getGhostStates()
  ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  numberOfCapsulesLeft = len(currentGameState.getCapsules())
  # ghostPositions = []
  # agentNum = currentGameState.getNumAgents()
  # for ghostIndex in range(1, agentNum):
  #   ghostPositions.append()
  nearestFoodDistance = float('inf')
  for foodPosition in foodList:
    distance = util.manhattanDistance(pacmanPosition, foodPosition)
    if distance > nearestFoodDistance:
      # nearestFoodPosition = foodPosition
      nearestFoodDistance = distance

  nearestCapsulesDistance = float('inf')
  for capsulesPosition in capsulesList:
    distance = util.manhattanDistance(pacmanPosition, capsulesPosition)
    if distance > nearestCapsulesDistance:
      # nearestCapsulesPosition = capsulesPosition
      nearestCapsulesDistance = distance

  # foodMeanPosition = [sum(y) / len(y) for y in zip(*foodList)]
  # distToFoodMeanPosition = util.manhattanDistance(pacmanPosition, foodMeanPosition)
  
  

  "=====eval1: score of currentGameState and weighted by distance to nearest food====="
  eval[0] = currentGameState.getScore() + (10*(2**(-nearestFoodDistance))) + (1000*(2**(-nearestCapsulesDistance)))

  "=====eval2: Expected value of scores of all legal successorGameState====="
  # for action in pacmanLegalActions:
  #   successor = currentGameState.generateSuccessor(0, action)
  #   eval[1] += successor.getScore()/float(len(pacmanLegalActions))

  "=====eval3: negative value of tatal times that visiting this current point====="
  
  # eval[2] = - self.visitedPoints.count(agentPositions(0))

  "=====eval4: evaluated by total weighted value of the distance between pacman to each ghost====="
  nearestGhostDistance = float('inf')
  for ghostIndex in range(1, currentGameState.getNumAgents()):
      distance = util.manhattanDistance(pacmanPosition, currentGameState.getGhostPosition(ghostIndex))
      # if distance < nearestGhostDistance:
      #   nearestGhostDistance = distance
      if ghostScaredTimes[ghostIndex-1] > distance:
        eval[3] +=  (30*(2**(-distance)))
      else: #ghostScaredTimes[index] == 0:
        eval[3] -=  (50*(2**(-distance)))
  
  "=====eval5: evaluated by number of capsules left====="
  eval[4] -= 50*numberOfCapsulesLeft

  totalEval = 0
  for index in range(len(eval)):
    totalEval += weight[index] * eval[index]
  # print 'totalEval: ', totalEval
  return totalEval

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  # def getAction(self, gameState):
  #   """
  #     Returns an action.  You can use any method you want and search to any depth you want.
  #     Just remember that the mini-contest is timed, so you have to trade off speed and computation.

  #     Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
  #     just make a beeline straight towards Pacman (or away from him if they're scared!)
  #   """
  #   "*** YOUR CODE HERE ***"
  #   return 'Stop'
  visitedPoints = []
  def miniconstestEvaluationFunction(self, currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
  
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    
    "=====Some useful information of currentGameState====="
    pacmanPosition = currentGameState.getPacmanPosition()
    # pacmanLegalActions = currentGameState.getLegalActions(0)
    capsulesList = currentGameState.getCapsules()
    foods = currentGameState.getFood()
    foodList = foods.asList()
    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  
    "nearest food distance"
    nearestFoodDistance = float('inf')
    for foodPosition in foodList:
      distance = util.manhattanDistance(pacmanPosition, foodPosition)
      if distance > nearestFoodDistance:
        # nearestFoodPosition = foodPosition
        nearestFoodDistance = distance
  
    "nearest capsule distance"
    nearestCapsulesDistance = float('inf')
    for capsulesPosition in capsulesList:
      distance = util.manhattanDistance(pacmanPosition, capsulesPosition)
      if distance > nearestCapsulesDistance:
        # nearestCapsulesPosition = capsulesPosition
        nearestCapsulesDistance = distance
    "====================================================="
  
    evaluation = 0
    "times the state being visited"
    evaluation -= self.visitedPoints.count(pacmanPosition)
  
    "number of food left"
    evaluation -= len(foodList)
  
    "number of capsules left"
    if any(ghostScaredTimes):
      evaluation += 20*len(capsulesList)
    else:
      evaluation -= 20*len(capsulesList)
  
    "distance to nearest food & distance to nearest capsule"
    evaluation += currentGameState.getScore() + (10*(2**(-nearestFoodDistance))) + (50*(2**(-nearestCapsulesDistance)))
  
    "=====evaluated by total weighted value of the distance between pacman to each ghost====="
    nearestGhostDistance = float('inf')
    for ghostIndex in range(1, currentGameState.getNumAgents()):
      distance = util.manhattanDistance(pacmanPosition, currentGameState.getGhostPosition(ghostIndex))
      # if distance < nearestGhostDistance:
      #   nearestGhostDistance = distance
      if ghostScaredTimes[ghostIndex-1] > distance:
        evaluation +=  (30*(2**(-distance)))
      else: #ghostScaredTimes[index] == 0:
        evaluation -=  (50*(2**(-distance)))

    "is lose? & is win?"
    if currentGameState.isLose():
      evaluation -= 2000
    elif currentGameState.isWin(): 
      evaluation += 2000

    return evaluation

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    "*** YOUR CODE HERE ***"
    depth = 4
    "expectimax"
    # def maxValue(gameState, agentIndex, depth, ghostNum):
    #   value = float('-inf')
    #   for action in gameState.getLegalActions(agentIndex):
    #     if action != 'Stop':
    #       value = max(value, expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum) - self.visitedPoints.count(gameState.getPacmanPosition()))
    #   return value

    # def expValue(gameState, agentIndex, depth, ghostNum):
    #   value = 0
    #   legalActions = gameState.getLegalActions(agentIndex)
    #   for action in legalActions:
    #     if action != 'Stop':
    #       if agentIndex == ghostNum:
    #         value = value + expectimax(gameState.generateSuccessor(agentIndex, action), 0, depth-1, ghostNum)/float(len(legalActions))
    #       else: #agentIndex < ghostNum
    #         value = value + expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum)/float(len(legalActions))
    #   return value

    # def expectimax(gameState, agentIndex, depth, ghostNum):
    #   if gameState.isWin() or gameState.isLose() or depth == 0:
    #     return self.miniconstestEvaluationFunction(gameState)
    #   if agentIndex == 0:
    #     return maxValue(gameState, agentIndex, depth, ghostNum)
    #   if agentIndex > 0:
    #     return expValue(gameState, agentIndex, depth, ghostNum)
    
    # bestValue = float('-inf')
    # bestAction = Directions.STOP
    # for action in gameState.getLegalActions(0):
    #   if action != 'Stop':
    #     value = expectimax(gameState.generateSuccessor(0, action), 1, depth, gameState.getNumAgents()-1)
    #     # print 'value: ', value
    #     if value > bestValue:
    #       bestValue = value
    #       bestAction = action
    # self.visitedPoints.append(gameState.getPacmanPosition())
    # return bestAction

    "alph-beta prunning"
    def maxValue(gameState, agentIndex, depth, ghostNum, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.miniconstestEvaluationFunction(gameState)
      value = float('-inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          value = max(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum, alpha, beta))
          if value >= beta:
            return value
          alpha = max(alpha, value)
      return value

    def minValue(gameState, agentIndex, depth, ghostNum, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.miniconstestEvaluationFunction(gameState)
      value = float('inf')
      for action in gameState.getLegalActions(agentIndex):
        if action != 'Stop':
          if agentIndex == ghostNum:
            value = min(value, maxValue(gameState.generateSuccessor(agentIndex, action), 0, depth-1, ghostNum, alpha, beta))
          else:
            value = min(value, minValue(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth, ghostNum, alpha, beta))
          if value <= alpha:
            return value
          beta = min(beta, value)
      return value

    
    alpha = float('-inf')
    beta = float('inf')
    ghostNum = gameState.getNumAgents()-1
    value = float('-inf')
    bestValue = float('-inf')
    bestAction = Directions.STOP

    for action in gameState.getLegalActions(0):
      if action != 'Stop':
        value = max(value, minValue(gameState.generateSuccessor(0, action), 1, depth, ghostNum, alpha, beta))
        if value >= beta:
          return action
        alpha = max(alpha, value)
        if value > bestValue:
          bestValue = value
          bestAction = action
    self.visitedPoints.append(gameState.getPacmanPosition)
    return bestAction