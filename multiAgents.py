# multiAgents.py
# --------------
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
        oldFoodNum = currentGameState.getNumFood()
        newFoodNum = successorGameState.getNumFood()
        flag=1
        if oldFoodNum-newFoodNum != 0:
            flag=0
        minDis = min([manhattanDistance(GhostPos.getPosition(),newPos) for GhostPos in newGhostStates])
        thex=0
        they=0

        #print("reached")
        #cap = successorGameState.getCapsules()
        if successorGameState.getNumFood()!=0:
            for x in range(newFood.width):
                for y in range(newFood.height):
                    if newFood[x][y] == True :
                        thex=x
                        they=y
                        break
            xy=[thex,they]
            disToFood=manhattanDistance(xy,newPos)
        else:
            disToFood=0

        #disToFood=0

        """
        if len(cap)!=0:
            minCap = min([manhattanDistance(capPos,newPos) for capPos in cap])
        else:
            minCap = 0
            """

        if all(newScared for newScared in newScaredTimes)>0:
            return -(1000)*newFood.count()-10*disToFood-minDis
        if minDis<5:
            return -newFood.count()-0.01*disToFood+minDis
        return -newFood.count()-0.01*disToFood+5
        "return successorGameState.getScore()"

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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        result = self._maxvalue(gameState)

        return result[1][1]

    # Max value expand
    def _maxvalue(self, state, agent_index=0, depth=1):

        actions = state.getLegalActions(agent_index)
        successors = [tuple((state.generateSuccessor(agent_index, action), action)) for action in actions]

        """
        print "---------"
        print "max"
        print "depth: %d" % depth
        print state
        print actions
        """

        # The return value is a list of tuple of (minvalue, the state of this minvalue)
        result = [tuple((self._minvalue(s[0], 1, depth)[0], s)) for s in successors]

        # If no action can be taken, this means a terminal condition occurs,
        # return score directly
        if len(result) is 0:
            return tuple((state.getScore(), state))

        max_heuristic = max(result, key=lambda x: x[0])[0]
        result = [r for r in result if r[0] == max_heuristic]

        if len(result) is 1:
            return result[0]
        else:
            return filter(lambda x: x[1][1] is not Directions.STOP, result)[0]

    # Min value expand
    def _minvalue(self, state, agent_index, depth):

        actions = state.getLegalActions(agent_index)
        successors = [tuple((state.generateSuccessor(agent_index, action), action)) for action in actions]

        """
        print "---------"
        print "min"
        print "depth: %d" % depth
        print state
        print actions
        """

        if agent_index < state.getNumAgents() - 1:
            result = [tuple((self._minvalue(s[0], agent_index + 1, depth)[0], s)) for s in successors]
        else:
            if depth >= self.depth:
                return tuple((state.getScore(), state))
            else:
                result = [tuple((self._maxvalue(s[0], 0, depth + 1)[0], s)) for s in successors]


        if len(result) is 0:
            return tuple((state.getScore(), state))

        min_heuristic = min(result, key=lambda x: x[0])[0]
        result = [r for r in result if r[0] == min_heuristic]


        if len(result) is 1:
            return result[0]
        else:
            return filter(lambda x: x[1][1] is not Directions.STOP, result)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction



