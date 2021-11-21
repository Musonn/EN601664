# Boyang Zhang, Muchen Li
# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState()    # start state
    visited = []
    s = util.Stack() # initialize stack
    s.push((startState, [])) # item example: ((5,5),'w')

    while s.isEmpty() is False:
        (parent,path) = s.pop()
        if parent not in visited:
            if problem.isGoalState(parent) is True:
                #print('path is', path)
                return path
                #return translate(path)   # till here the path is finished. Next is to translate into correct format
            visited.append(parent)
            for child in problem.getSuccessors(parent):
                edge = child[0]
                s.push((edge,path + [child[1]]))

    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState()    # start state
    visited = []
    q = util.Queue() # initialize queue
    q.push((startState, [])) # item example: ((5,5),['w'])

    while q.isEmpty() is False:
        (parent,path) = q.pop()
        #print('cur state is', parent)
        if parent not in visited:
            if problem.isGoalState(parent) is True:
                return path
            visited.append(parent)
            for child in problem.getSuccessors(parent):
                edge = child[0]
                q.push((edge,path + [child[1]]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()    # start state
    visited = []
    pq = util.PriorityQueue() # initialize stack
    pq.push((startState, [], 0), 0) # (state, priority) example: ((5,5),['w'], 0)
                                    # priority is a parameter used by priority queue

    while pq.isEmpty() is False:
        (parent, path, cost) = pq.pop()
        #print(parent,total_cost)
        if parent not in visited:
            if problem.isGoalState(parent) is True:
                return path
            visited.append(parent)
            for child in problem.getSuccessors(parent):
                edge = child[0]
                direction = path + [child[1]]
                total_cost = cost + child[2]
                pq.push((edge,direction,total_cost), total_cost)
                #print(edge, total_cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startNode = problem.getStartState()    # start state
    start_total_cost = heuristic(startNode,problem)+0
    visited = []
    pq = util.PriorityQueue() # initialize stack
    pq.push((startNode, [], 0),start_total_cost) # (state, priority): Here state is a tuple containing the node, path and cost. example: ((5,5),['w'], 0)
                                    # priority is a parameter used by priority queue
    while pq.isEmpty() is False:
        parent, path, cost = pq.pop()
        #print(parent, cost)
        if parent not in visited:
            if problem.isGoalState(parent) is True:
                return path
            visited.append(parent)
            for child in problem.getSuccessors(parent):
                edge = child[0]
                direction = path + [child[1]]
                if edge not in visited:
                    total_cost = cost + child[2] 
                    thepriority = total_cost + heuristic(edge, problem)
                    pq.push((edge,direction,total_cost),thepriority)

def translate(path):
    '''
    This function translate path-in-string to path-in-directions
    '''
    from game import Directions
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    s = Directions.SOUTH
    path2 = []

    for i in path:
        if i == 'West': path2.append(w)
        elif i == 'North': path2.append(n)
        elif i == "East": path2.append(e)
        else: path2.append(s)
    return path2

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
