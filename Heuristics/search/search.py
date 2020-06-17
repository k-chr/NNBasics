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
from util import *
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

def DFS(problem, node, visited, path = [])->list:
    visited.add(node)
    if problem.isGoalState(node) == True:
        return path
    for next_node, direction, _ in problem.getSuccessors(node):
        if next_node not in visited:
            s = path.copy()
            s.append(direction)
            p = DFS(problem, next_node, visited, s)
            if len(p) > 0:
                return p
    return list()       
                
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
    visited = set()
    start_node = problem.getStartState()
    results = DFS(problem, start_node, visited, [])
    return results
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    q = Queue()
    start_node = problem.getStartState()
    visited = set()
    results = []
    results.append([])
    q.push(start_node)
    while not q.isEmpty():
        node = q.pop()
        path_to_node = results.pop(0)
        if problem.isGoalState(node):
            return path_to_node
        if node not in visited:
            visited.add(node)
            for next_node, direction, _ in problem.getSuccessors(node):
                path = list(path_to_node)
                path.append(direction)
                q.push(next_node)
                results.append(path)
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    q = PriorityQueue()
    start_node = problem.getStartState()
    visited = set()
    results = PriorityQueue()
    results.push([], 0)
    q.push((start_node, 0), 0)
    while not q.isEmpty():
        node, weight = q.pop()
        path_to_node = results.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in visited:
            visited.add(node)
            for next_node, direction, node_weight in problem.getSuccessors(node):
                path = list(path_to_node)
                path.append(direction)
                q.push((next_node, node_weight + weight), node_weight + weight)
                results.push(path, node_weight + weight)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    q = PriorityQueue()
    start_node = problem.getStartState()
    visited = set()
    results = PriorityQueue()
    results.push([], 0)
    q.push((start_node, 0), 0)
    while not q.isEmpty():
        node, weight = q.pop()
        path_to_node = results.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in visited:
            visited.add(node)
            for next_node, direction, node_weight in problem.getSuccessors(node):
                path = list(path_to_node)
                path.append(direction)
                q.push((next_node, node_weight + weight), node_weight + weight + heuristic(next_node, problem))
                results.push(path, node_weight + weight + heuristic(next_node, problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
