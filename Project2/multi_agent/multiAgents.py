# Boyang Zhang, Muchen Li
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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # state action
        # distance and value
        availableFood = newFood.asList()
        #print(newCapsules)
        newGhostPos = successorGameState.getGhostPositions()
        #print(newGhostPos)
        # find nearest food
        closestFood = float("inf")
        for f in availableFood:
            closestFood = min(closestFood, manhattanDistance(newPos, f))
        for g in newGhostPos:
            if manhattanDistance(newPos, g) < 5:
                final_score = -float("inf")
            else:
                final_score = successorGameState.getScore() + 1.0/closestFood # reciporal successorGameState.getScore() * 1.0/closestFood (not work)
        return final_score

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
        self.evaluationFunctionName = evalFn    # This line is added by Muchen
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
        # Collect legal moves and successor states

        def getValue(gameState, agentIndex, depth):  # depth - depth limit search
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if gameState.isLose() or gameState.isWin() or depth is self.depth:
                val = self.evaluationFunction(gameState)
            elif agentIndex is 0:
                val = max_value(gameState, agentIndex, depth)
            else:
                val = min_value(gameState, agentIndex, depth)
            return val

        def max_value(gameState, agentIndex, depth):
            # initialize
            v = ['initialize', -float("inf")]
            legal_actions = gameState.getLegalActions(agentIndex)
            if len(legal_actions) is 0:
                return self.evaluationFunction(gameState)
            else:
                # loop
                for act in gameState.getLegalActions(agentIndex):
                    current_state = gameState.generateSuccessor(agentIndex, act)
                    current_value = getValue(current_state, agentIndex + 1, depth)

                    if type(current_value) is not list:
                        current_value = ['', current_value]
                    # get v
                    if current_value[1] > v[1]:
                        v = [act, current_value[1]]
            return v

        def min_value(gameState, agentIndex, depth):
            # initialize
            v = ['initialize', float("inf")]
            legal_actions = gameState.getLegalActions(agentIndex)
            if len(legal_actions) is 0:
                return self.evaluationFunction(gameState)
            else:
                # loop
                for act in legal_actions:
                    current_state = gameState.generateSuccessor(agentIndex, act)
                    current_value = getValue(current_state, agentIndex + 1, depth)

                    if type(current_value) is not list:
                        current_value = ['', current_value]
                    # get v
                    if current_value[1] < v[1]:
                        v = [act, current_value[1]]
            return v

        return getValue(gameState, 0, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def getValue(gameState, agentIndex, depth, alpha, beta):  # depth - depth limit search
            if agentIndex >= gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if gameState.isLose() or gameState.isWin() or depth is self.depth:
                val = self.evaluationFunction(gameState)
            elif agentIndex is 0:
                val = max_value(gameState, agentIndex, depth, alpha, beta)
            else:
                val = min_value(gameState, agentIndex, depth, alpha, beta)
            return val

        def max_value(gameState, agentIndex, depth, alpha, beta):
            # initialize
            v = ['initialize', -float("inf")]
            legal_actions = gameState.getLegalActions(agentIndex)
            if len(legal_actions) is 0:
                return self.evaluationFunction(gameState)
            else:
                # loop
                for act in gameState.getLegalActions(agentIndex):
                    current_state = gameState.generateSuccessor(agentIndex, act)
                    current_value = getValue(current_state, agentIndex + 1, depth, alpha, beta)

                    if type(current_value) is not list:
                        current_value = ['', current_value]
                    # get v
                    if current_value[1] > v[1]:
                        v = [act, current_value[1]]
                    # prune
                    if v[1] > beta:  # not equality
                        return [act, current_value[1]]
                    else:
                        alpha = max(alpha, v[1])
            return v

        def min_value(gameState, agentIndex, depth, alpha, beta):
            # initialize
            v = ['initialize', float("inf")]
            legal_actions = gameState.getLegalActions(agentIndex)
            if len(legal_actions) is 0:
                return self.evaluationFunction(gameState)
            else:
                # loop
                for act in legal_actions:
                    current_state = gameState.generateSuccessor(agentIndex, act)
                    current_value = getValue(current_state, agentIndex + 1, depth, alpha, beta)

                    if type(current_value) is not list:
                        current_value = ['', current_value]
                    # get v
                    if current_value[1] < v[1]:
                        v = [act, current_value[1]]
                    # prune
                    if v[1] < alpha:  # not equality
                        return [act, current_value[1]]
                    else:
                        beta = min(beta, v[1])
            return v

        return getValue(gameState, 0, 0, -float("inf"), float("inf"))[0]

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
        return self.getValue(gameState, None, 0)[0]

    def getValue(self, gameState, action, index):
        legal_actions = gameState.getLegalActions(index)
        chanceList=[]
        dictionary ={}

        if self.depth == 0 or gameState.isWin():
            return [action, self.evaluationFunction(gameState)]

        isPackman = True if index == 0 else False

        if isPackman:   # packman node aka max node
            for act in legal_actions:
                child_state = gameState.generateSuccessor(index, act)
                dictionary[self.getValue(child_state, act, 1)[1]] = self.getValue(child_state, act, 1)[0]
            if len(dictionary) == 0: return ['Stop',-1e10]   # not sure why dictionary could be empty
            Action = dictionary[max(dictionary.keys())]
            value = max(dictionary.keys())
            return [Action, value]
        
        else:   # chance node
            for act in legal_actions:
                child_state = gameState.generateSuccessor(index, act)
                if index < gameState.getNumAgents() - 1:    # check if more than one ghost
                    a = ExpectimaxAgent(depth= self.depth)
                    chanceList += [self.getValue(child_state, act, index+1)[1]]
                else:
                    a = ExpectimaxAgent(evalFn = self.evaluationFunctionName, depth= self.depth-1)  # if not, pass to pacman
                    chanceList += [a.getValue(child_state, act, 0)[1]]
                
            try: expectation = sum(chanceList) / len(legal_actions) # abnormality handling
            except ZeroDivisionError: expectation = -1e10   # -inf not work
            return [action, expectation]

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
        This evaluation function takes advantage of that in q1 which is written by Boyang.
        The function uses three strategies: go for the clostest food; avoid the ghost; go for the ghost if pacman devoured a pellet.
        -Muchen
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curCapsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    availableFood = curFood.asList()
    GhostPos = currentGameState.getGhostPositions()

    # find nearest food
    closestFood = float("inf")
    closestCapsule = float("inf")

    for c in curCapsules:
        closestCapsule = min(closestCapsule, manhattanDistance(curPos, c))
    for f in availableFood:
        closestFood = min(closestFood, manhattanDistance(curPos, f))
    for g in GhostPos:
        if manhattanDistance(curPos, g) < 5 and not ScaredTimes:
            final_score = -float("inf")
        elif ScaredTimes:
            final_score = currentGameState.getScore() + 1/(1+manhattanDistance(curPos, g))  # adding 1 to avoid ZerroDivisionError
        else:
            final_score = currentGameState.getScore() + 1.0/closestFood + 2.0/closestCapsule
    return final_score

# Abbreviation
better = betterEvaluationFunction

