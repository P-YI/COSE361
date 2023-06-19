# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

from game import Directions

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""

def myHeuristic(position, problem, info={}):
    from math import pi
    xy1 = position
    xy2 = problem.goal
    return pi * 0.5 * (((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])** 2)**0.5)

def createAgents(num_pacmen, agent='ClosestDotAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    num_pacmen = 0

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"

        # end가 false일 경우 정지.
        if self.end:
            return Directions.STOP

        # Index 업데이트.
        elif self.curIndex +1 < len(self.nextIndex):
            self.curIndex += 1
            return self.nextIndex[self.curIndex]

        else:
            # 다음 dot으로의 이동.
            self.nextIndex = list((self.pathToDots(state)))

            # Index 업데이트.
            if len(self.nextIndex) != 0:
                self.curIndex = 0
                return self.nextIndex[self.curIndex]
            
            # len(self.nextIndex)가 0이면 end를 True로 바꾸고 정지.
            else:
                self.end = True
                return Directions.STOP

        # raise NotImplementedError()

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"

        # 시작 전 Index 초기화. 초기화는 처음 한번만.
        MyAgent.num_pacmen = MyAgent.num_pacmen +1
        self.end = False
        self.curIndex = 0
        self.nextIndex = []

        # raise NotImplementedError()

    def pathToDots(self, gameState):

        # 각종 search algorithm 테스트. 가장 성능이 좋은 BFS 선택.

        problem = AnyFoodSearchProblem(gameState, self.index)

        # return myDFS(problem)
        return myBFS(problem)
        # return myUCS(problem)
        # return myASTAR(problem)

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        pacmanCurrent = [problem.getStartState(), [], 0]
        visitedPosition = set()
        # visitedPosition.add(problem.getStartState())
        fringe = util.PriorityQueue()
        fringe.push(pacmanCurrent, pacmanCurrent[2])
        while not fringe.isEmpty():
            pacmanCurrent = fringe.pop()
            if pacmanCurrent[0] in visitedPosition:
                continue
            else:
                visitedPosition.add(pacmanCurrent[0])
            if problem.isGoalState(pacmanCurrent[0]):
                return pacmanCurrent[1]
            else:
                pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
            Successor = []
            for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
                if item[0] not in visitedPosition:
                    pacmanRoute = pacmanCurrent[1].copy()
                    pacmanRoute.append(item[1])
                    sumCost = pacmanCurrent[2]
                    Successor.append([item[0], pacmanRoute, sumCost + item[2]])
            for item in Successor:
                fringe.push(item, item[2])
        return pacmanCurrent[1]

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        if self.food[x][y] == True:
            return True
        
        return False



def myDFS(problem):
    from util import Stack

    start = problem.getStartState()
    stack = Stack()
    stack.push([start, 0, []])
    path = []
    visited = []

    if problem.isGoalState(start):
        return []

    while(True):
        if stack.isEmpty():
            return []
        
        [node, cost, path] = stack.pop()

        if problem.isGoalState(node):
            return path

        if not node in visited:

            visited.append(node)
            successor = problem.getSuccessors(node)

            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path]
                stack.push([child_node, new_cost, new_path])


def myBFS(problem):
    from util import Queue

    start = problem.getStartState()
    queue = Queue()
    queue.push([start, 0, []])
    path = []
    visited = []

    if problem.isGoalState(start):
        return []

    while(True):
        if queue.isEmpty():
            return []
        
        [node, cost, path] = queue.pop()

        if problem.isGoalState(node):
            return path

        if not node in visited:

            visited.append(node)
            successor = problem.getSuccessors(node)

            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path]
                queue.push([child_node, new_cost, new_path])


def myUCS(problem):
    from util import PriorityQueue

    start = problem.getStartState()
    queue = PriorityQueue()
    priority = 0
    queue.push([start, 0, []], priority)
    path = []
    visited = []

    if problem.isGoalState(start):
        return []

    while(True):
        if queue.isEmpty():
            return []
        
        [node, cost, path] = queue.pop()

        if problem.isGoalState(node):
            return path

        if not node in visited:

            visited.append(node)
            successor = problem.getSuccessors(node)

            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path, ]
                queue.push([child_node, new_cost, new_path])


def myASTAR(problem, heuristic=myHeuristic):
    from util import PriorityQueue

    start = problem.getStartState()
    queue = PriorityQueue()
    priority = 0
    queue.push([start, 0, []], priority)
    path = []
    visited = []

    if problem.isGoalState(start):
        return []

    while(True):
        if queue.isEmpty():
            return []
        
        [node, cost, path] = queue.pop()

        if problem.isGoalState(node):
            return path

        if not node in visited:

            visited.append(node)
            successor = problem.getSuccessors(node)

            for child_node, child_path, child_cost in successor :
                new_cost = cost + child_cost
                new_path = path + [child_path, ]
                queue.push([child_node, new_cost, new_path], new_cost + heuristic(child_node, problem))

