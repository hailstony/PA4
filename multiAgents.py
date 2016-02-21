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
import random, util, sys

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
        return self._maxvalue(gameState)[1]

    # Max value expand
    # The return value of the function will be a tuple
    # (max_score, action_to_the_state_of_max_score)
    def _maxvalue(self, state, agent_index=0, depth=1):

        # If the maximum depth is exceeded, return the evaluted score
        # Since no action is taken, the place for the action will be None.
        # However, this won't affect, because it won't be used.
        if depth > self.depth:
            return tuple((self.evaluationFunction(state), None))

        # Get all possible actions for this agent,
        # and generate a list of tuple of (successor_game_state, action_lead_to_this_successor)
        actions = state.getLegalActions(agent_index)
        successors = [tuple((state.generateSuccessor(agent_index, action), action)) for action in actions]

        # The return value is a list of tuple of (minvalue, action)
        # The next agent for maxvalue will always be Ghost No.1
        result = [tuple((self._minvalue(s[0], 1, depth)[0], s[1])) for s in successors]

        # If no action can be taken, this means a terminal condition occurs,
        # return score directly
        if len(result) is 0:
            return tuple((self.evaluationFunction(state), None))

        max_heuristic = max(result, key=lambda x: x[0])[0]          # Max value
        result = [r for r in result if r[0] == max_heuristic]       # Get the lists of the largest value

        return result[0]

    # Min value expand
    # The return value of the function will be a tuple
    # (min_score, action_to_the_state_of_min_score)
    def _minvalue(self, state, agent_index, depth):

        # Get all possible actions for this agent,
        # and generate a list of tuple of (successor_game_state, action_lead_to_this_successor)
        actions = state.getLegalActions(agent_index)
        successors = [tuple((state.generateSuccessor(agent_index, action), action)) for action in actions]

        # If this is not the last ghost, run minvalue again on the next ghost
        # The return value is a list of tuple of (minvalue, action)
        if agent_index < state.getNumAgents() - 1:
            result = [tuple((self._minvalue(s[0], agent_index + 1, depth)[0], s[1])) for s in successors]
        # If this is the last ghost, run maxvalue on the pacman, add the depth
        # The return value is a list of tuple of (maxvalue, action)
        else:
            result = [tuple((self._maxvalue(s[0], 0, depth + 1)[0], s[1])) for s in successors]

        # If a terminal condition occurs, return directly
        if len(result) is 0:
            return tuple((self.evaluationFunction(state), None))

        min_heuristic = min(result, key=lambda x: x[0])[0]          # Min value
        result = [r for r in result if r[0] == min_heuristic]       # The list of succesor state of min value

        return result[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self._maxvalue(gameState, 0, 1, -sys.maxint - 1, sys.maxint)[1]

    # Max value expand
    # The return value of the function will be a tuple
    # (max_score, action_to_the_state_of_max_score)
    def _maxvalue(self, state, agent_index, depth, alpha, beta):

        # If maximum depth is exceeded, return the score directly
        if depth > self.depth:
            return tuple((self.evaluationFunction(state), None))

        v = -sys.maxint - 1     # Use a tmp var to get the max score
        a = None                # Use a tmp var to get the action to the max score

        actions = state.getLegalActions(agent_index)       # Possible actions

        # If no possbile actions are possible, return directly
        if len(actions) is 0:
            return tuple((self.evaluationFunction(state), None))

        # Do not call unnecessary generateSuccessor
        for action in actions:
            # Generate succesor and its score
            successor = tuple((state.generateSuccessor(agent_index, action), action))
            score = self._minvalue(successor[0], 1, depth, alpha, beta)

            # Store the tmp largest value and action accordingly
            if v < score[0]:
                v = score[0]
                a = successor[1]

            # Store alpha to the tmp largest value also
            if alpha < v:
                alpha = v

            # If the tmp maximum exceeds the tmp minimum on the upper level, return directly
            # The logical here is simple:
            # Any return value will be larger than this tmp largest value,
            # However, since there's a smaller tmp smaller value on the upper level
            # this tmp largest value will never selected, so just prune
            if beta < v:
                return tuple((v, a))

        return tuple((v, a))

    # Min value expand
    # The return value of the function will be a tuple
    # (min_score, action_to_the_state_of_min _score)
    def _minvalue(self, state, agent_index, depth, alpha, beta):
        v = sys.maxint      # Tmp minimum value
        a = None            # action to this tmp minimum value

        actions = state.getLegalActions(agent_index)    # Possible moves

        # If no possible moves, return directly
        if len(actions) is 0:
            return tuple((self.evaluationFunction(state), None))

        for action in actions:
            successor = tuple((state.generateSuccessor(agent_index, action), action))   # Succesor state

            # If it's last ghost agent, evaluate on the pacman
            # Else, evaluate on the next ghost
            if agent_index < state.getNumAgents() - 1:
                score = self._minvalue(successor[0], agent_index + 1, depth, alpha, beta)
            else:
                score = self._maxvalue(successor[0], 0, depth + 1, alpha, beta)

            # Store the tmp minimum value and the action accordingly
            if v > score[0]:
                v = score[0]
                a = successor[1]

            # Store the tmp minimum on beta too
            if beta > v:
                beta = v

            # If the upper level is still a minimum, alpha won't be changed,
            # So the pruning won't work.
            # If the upper level is a maximum, if the tmp minimum is less than
            # the tmp maximum of that maximum level, stop expanding
            if alpha > v:
                return tuple((v, a))

        return tuple((v, a))


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
        return self._maxvalue(gameState, 0, 1)[1]

    # Max value expand
    # The return value of the function will be a tuple
    # (max_score, action_to_the_state_of_max_score)
    def _maxvalue(self, state, agent_index=0, depth=1):

        # If the maximum depth is exceeded, return the evaluted score
        # Since no action is taken, the place for the action will be None.
        # However, this won't affect, because it won't be used.
        if depth > self.depth:
            return tuple((self.evaluationFunction(state), None))

        # Get all possible actions for this agent,
        # and generate a list of tuple of (successor_game_state, action_lead_to_this_successor)
        actions = state.getLegalActions(agent_index)

        # If no action can be taken, this means a terminal condition occurs,
        # return score directly
        if len(actions) is 0:
            return tuple((self.evaluationFunction(state), None))

        v = -sys.maxint - 1     # Use a tmp var to get the max score
        a = None                # Use a tmp var to get the action to the max score

        for action in actions:
            successor = tuple((state.generateSuccessor(agent_index, action), action))
            score = self._expvalue(successor[0], 1, depth)

            # Store the tmp largest value and action accordingly
            if v < score[0]:
                v = score[0]
                a = successor[1]

        return tuple((v, a))

    # expectation value expand
    # The return value of the function will be a tuple
    # (min_score, action_to_the_state_of_min_score)
    def _expvalue(self, state, agent_index, depth):

        # Get all possible actions for this agent,
        # and generate a list of tuple of (successor_game_state, action_lead_to_this_successor)
        actions = state.getLegalActions(agent_index)

        # If no action can be taken, this means a terminal condition occurs,
        # return score directly
        if len(actions) is 0:
            return tuple((self.evaluationFunction(state), None))

        sum = 0.0       # sum to count the total score

        for action in actions:
            successor = tuple((state.generateSuccessor(agent_index, action), action))

            if agent_index < state.getNumAgents() - 1:
                score = self._expvalue(successor[0], agent_index + 1, depth)
            else:
                score = self._maxvalue(successor[0], 0, depth + 1)

            sum += score[0]

        # Suppose the estimation of probability to each direction is the same
        avg = float(sum) / float(len(actions))

        return tuple((avg, None))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The intension of the evalutaion is to clear the map, where every possible place is BLANK/SPACE.
      1. So give the BLANK/SPACE really high positive value to evaluate
      2. Also give food relatively high positive vaule to evaluate to approach.
            a. The value of food should increase when the number of foods decreases to dominate the affection
               of walls and ghost
      3. Give walls next to the pacman a low negative value to get out of the place with a lot of walls
      4. Give a ghost a relatively high negative value
            a. Give it really high negative value when it is near the pacman
      5. Give Capsule the value which is a little bit higher than the food to make it more attractive
      6. After eating a capsule, give the ghost a really high positive value instead to chase.
    """
    "*** YOUR CODE HERE ***"

    # Define keys, and their id in the map
    CAP = "Cap"
    FOOD = "Food"
    SPACE = "Space"
    WALL = "Wall"
    GHOST = "Ghost"

    ITEMS = {
        CAP: 2,
        FOOD: 1,
        SPACE: 0,
        WALL: -1,
        GHOST: -2
    }

    # Get foods, walls, capsules, ghosts and packman's location
    foods = currentGameState.getFood()
    walls = currentGameState.getWalls()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostPositions()
    pos = currentGameState.getPacmanPosition()

    # Combine them into a single map
    map = [[ITEMS[SPACE] for f in food] for food in foods]

    for i, food in enumerate(foods):
        for j, f in enumerate(food):
            if foods[i][j] is True:
                map[i][j] = ITEMS[FOOD]
            elif walls[i][j] is True:
                map[i][j] = ITEMS[WALL]

    for loc in capsules:
        map[loc[0]][loc[1]] = ITEMS[CAP]

    for loc in ghosts:
        map[int(loc[0])][int(loc[1])] = ITEMS[GHOST]

    food_num = currentGameState.getNumFood() + len(capsules)    # Number of foods in map
    # In case the food becomes 0
    if food_num == 0.0:
        food_num = 1.0

    # Define FOOD, WALL and GHOST's constant
    FOOD_SCORE = 100.0
    WALL_SCORE = -0.45
    GHOST_SCORE = -3.0

    # MAX STEPs can go in map
    MAX_STEP = float(util.manhattanDistance((0, 0), (len(map) - 1, len(map[0]) - 1)))

    # Store each category's values separately
    values = dict()
    values[FOOD] = 0
    values[CAP] = 0
    values[SPACE] = 0
    values[GHOST] = 0
    values[WALL] = 0

    # Go through each element in map
    for i in xrange(len(map)):
        for j in xrange(len(map[0])):

            # If it is BLANK, give it really high value (which must be higher than any possible food value)
            # Otherwise packman won't eat it, cuz don't eat will have a higher value.
            # And space should have no relationship with the distance
            if map[i][j] == ITEMS[SPACE]:
                values[SPACE] += FOOD_SCORE * MAX_STEP
            else:
                # Food's value will increase when the number of food decreases
                # So the food's change in values will dominate the change in values of WALL
                if map[i][j] == ITEMS[FOOD]:
                    tmp = FOOD_SCORE / food_num
                    item = FOOD
                # Give CAP a little bit higher value than food to make it more attractive
                elif map[i][j] == ITEMS[CAP]:
                    tmp = FOOD_SCORE / food_num * 0.55
                    item = CAP
                # Give WALL a penalty if there's a wall on the 4 possible directions
                elif map[i][j] == ITEMS[WALL]:
                    if (i == pos[0] and j == pos[1] - 1) or (i == pos[0] and j == pos[1] + 1) or \
                       (i == pos[0] - 1 and j == pos[1]) or (i == pos[0] + 1 and j == pos[1]):
                        tmp = WALL_SCORE
                    else:
                        tmp = 0
                    item = WALL
                elif map[i][j] == ITEMS[GHOST]:
                    tmp = GHOST_SCORE
                    item = GHOST

                # Distance from pacman to the element
                dis = float(util.manhattanDistance(pos, tuple((i, j))))
                v = (MAX_STEP - dis) * tmp

                if item == GHOST:
                    # Get the ghost's agent index to check the scaredTimer
                    for index in xrange(currentGameState.getNumAgents()):
                        if index == 0:
                            continue

                        p = currentGameState.getGhostPosition(index)
                        if int(p[0]) == i and int(p[1]) == j:
                            break

                    ghost_state = currentGameState.getGhostState(index)

                    # If a ghost can be eaten, give it really high positive value to chase
                    if ghost_state.scaredTimer > 0:
                        v = FOOD_SCORE * (MAX_STEP - 1)
                    # Otherwise, give it really high negative value to escape
                    elif dis < 2:
                        v *= 2000.0
                values[item] += v

    return sum([values[key] for key in values.keys()])


# Abbreviation
better = betterEvaluationFunction



