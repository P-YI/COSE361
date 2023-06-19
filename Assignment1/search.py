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

    # DataStructure로 Stack 선택. 
    # DFS는 한 경로의 깊이 끝까지 갔다가 다시 갈림길로 되돌아와 탐색을 반복해야 함.
    # 이에 후입 선출 구조(LIFO)인 Stack 활용.
    from util import Stack

    start = problem.getStartState()
    stack = Stack()
    stack.push([start, 0, []]) # state인 start를 return하는 방식으로 push.
    path = [] # start state부터 방문한 path.
    visited = [] # 방문한 state.

    # 초기 state가 goal인지 확인.
    if problem.isGoalState(start) : 
        return []

    while(True) : 

        # 해결 못한 경우. stack이 비었을 때 return.
        if stack.isEmpty() : 
            return []

        # 현재의 state에 대한 정보. stack에서 node, cost, path 가져옴.
        [node, cost, path] = stack.pop()

        # goal에 도착했을 경우, path를 return.
        if problem.isGoalState(node) :
            return path

        # 아직 node가 visited에 속하지 않은 경우,
        if not node in visited : 

            # visited에 node 추가.
            visited.append(node)
            # node에 대한 successor 가져옴.
            successor = problem.getSuccessors(node)

            # 자식 노드 우선 탐색. path를 다시 계산하고, 새롭게 state를 stack에 push.
            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path]
                stack.push([child_node, new_cost, new_path])
    
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # DataStructure로 Queue 선택. 
    # BFS는 순서대로 얕은 깊이부터 더 방문할 곳이 없을 떄까지 차례로 방문하는 방식.
    # 이에 선입 선출 구조(FIFO)인 Queue 활용.
    from util import Queue

    start = problem.getStartState()
    queue = Queue()
    queue.push([start, 0, []]) # state인 start를 return하는 방식으로 push.
    path = [] # start state부터 방문한 path.
    visited = [] # 방문한 state.

    # 초기 state가 goal인지 확인.
    if problem.isGoalState(start) : 
        return []

    while(True) : 

        # 해결 못한 경우. queue가 비었을 때 return.
        if queue.isEmpty() : 
            return []

        # 현재의 state에 대한 정보. queue에서 node, cost, path 가져옴.
        [node, cost, path] = queue.pop()

        # goal에 도착했을 경우, path를 return.
        if problem.isGoalState(node) :
            return path

        # 아직 node가 visited에 속하지 않은 경우,
        if not node in visited : 

            # visited에 node 추가.
            visited.append(node)
            # node에 대한 successor 가져옴.
            successor = problem.getSuccessors(node)

            # 자식 노드 탐색. path를 다시 계산하고, 새롭게 state를 queue에 push.
            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path]
                queue.push([child_node, new_cost, new_path])

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # DataStructure로 PriorityQueue 선택. 
    # UCS는 자식 노드로 가는 총 비용을 비교 후 가장 작은 비용이 드는 노드로 확장하는 방식.
    # 다만, 여기서 시작 노드 지정 후 다른 모든 노드에 대한 최단 경로를 파악함.
    # 이에 노드와 그 노드까지의 비용을 합쳐서 저장해야 하므로 PriorityQueue 활용.
    from util import PriorityQueue

    start = problem.getStartState()
    queue = PriorityQueue()
    priority = 0 # 비용 저장을 위한 priority 추가.
    queue.push([start, 0, []], priority) # state인 start를 return하는 방식으로 push.
    path = [] # start state부터 방문한 path.
    visitied = [] # 방문한 state.

    # 초기 state가 goal인지 확인.
    if problem.isGoalState(start) : 
        return []

    while(True) : 

        # 해결 못한 경우. queue가 비었을 때 return.
        if queue.isEmpty() : 
            return []

        # 현재의 state에 대한 정보. queue에서 node, cost, path 가져옴.
        [node, cost, path] = queue.pop()

        # goal에 도착했을 경우, path를 return.
        if problem.isGoalState(node) :
            return path

        # 아직 node가 visited에 속하지 않은 경우,
        if not node in visitied : 

            # visited에 node 추가.
            visitied.append(node)
            # node에 대한 successor 가져옴.
            successor = problem.getSuccessors(node)

            # 자식 노드 탐색. path를 다시 계산하고, 새롭게 state를 queue에 push.
            # path의 cost가 이전에 비해 더 적은지 확인.
            # 부모 노드보다 자녀 노드가 비용이 적다면, 업데이트.
            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path, ]
                queue.push([child_node, new_cost, new_path], new_cost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # DataStructure로 PriorityQueue 선택. 
    # aStar는 시작 노드와 도착 노드를 지정하여 최단 경로 파악.
    # heuristic 활용하여 알고리즘의 개선 가능. goal에 대한 정보 제공.
    # 자료 구조는 노드와 그 노드까지의 비용을 합쳐서 저장해야 하므로 PriorityQueue 활용.
    from util import PriorityQueue

    start = problem.getStartState()
    queue = PriorityQueue()
    priority = 0 # 비용 저장을 위한 priority 추가.
    queue.push([start, 0, []], priority) # state인 start를 return하는 방식으로 push.
    path = [] # start state부터 방문한 path.
    visitied = [] # 방문한 state.

    # 초기 state가 goal인지 확인.
    if problem.isGoalState(start) : 
        return []

    while(True) : 

        # 해결 못한 경우. queue가 비었을 때 return.
        if queue.isEmpty() : 
            return []

        # 현재의 state에 대한 정보. queue에서 node, cost, path 가져옴.
        [node, cost, path] = queue.pop()

        # goal에 도착했을 경우, path를 return.
        if problem.isGoalState(node) :
            return path

        # 아직 node가 visited에 속하지 않은 경우,
        if not node in visitied : 

            # visited에 node 추가.
            visitied.append(node)
            # node에 대한 successor 가져옴.
            successor = problem.getSuccessors(node)

            # 자식 노드 탐색. path를 다시 계산하고, 새롭게 state를 queue에 push.
            # path의 cost가 이전에 비해 더 적은지 확인.
            # 부모 노드보다 자녀 노드가 비용이 적다면, 업데이트.
            # 비용 계산에 heuristic 활용.
            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path, ]
                queue.push([child_node, new_cost, new_path], new_cost + heuristic(child_node, problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
