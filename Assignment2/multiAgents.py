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


from multiprocessing.sharedctypes import Value
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        # 초기값.
        foodD = []
        foodSucc = 0
        score = 0

        # 팩맨이 움직일 수 있는 다음 게임 상태,
        # 이게 terminal state 중 승리일 때 infinity 반환.
        if successorGameState.isWin() : 
            return float("inf")
        
        # 유령의 상태.
        for ghost in successorGameState.getGhostStates() : 

            # 거리로 manhattanDistance 활용.
            # newPos로 팩맨의 위치 정보 받아옴.
            # 팩맨과 유령의 최소 거리가 2보다 작은지 검사.
            # 이때 유령과 부딪힘을 방지하기 위해 -infinity 반환.
            if util.manhattanDistance(ghost.getPosition(), newPos) < 2 : 
                return float("-inf")
        
        # 음식의 상태.
        for food in list(newFood.asList()) : 

            # 팩맨과 음식의 거리를 manhattanDistace로 구함.
            foodD.append(util.manhattanDistance(food, newPos))

        # 현재 남아있는 음식이 다음 게임 상태의 음식보다 많을 경우, food successor에 점수 부여.
        if (currentGameState.getNumFood() > successorGameState.getNumFood()) : 
            foodSucc = 1000

        # 평가를 위한 점수 반환.
        # 다음 게임 상태의 점수에 food successor 더해줌.
        # 추가로 음식까지의 최소 거리를 배수로 뺀다.
        score = successorGameState.getScore() + foodSucc -10 * min(foodD)

        return score

        return successorGameState.getScore()

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Minimax search는 opponent가 충분히 영리할 때 최선의 선택을 하기 위한 방법
        # state-space search tree 기반
        # def value를 중심으로, terminal state이면 state의 utility 반환.
        # 만약 다음에 오는 agent가 MAX이면 def max-value, MIN이면 def min-value.

        v = -float("inf") # 초기 값을 -infinity로 지정. 최대값을 구하기 때문에.
        minimaxAction = Directions.STOP # 움직임을 정지 상태로 선언.
        
        # 가능한 움직임마다의 반복문 실행.
        for action in gameState.getLegalActions(0) : 
            newValue = self.value(gameState.generateSuccessor(0, action), 0, 1)

            # 최소값 중 최대값 구하기 위한 비교.
            if newValue > v : 
                v = newValue
                minimaxAction = action
        
        return minimaxAction

    # def value 구현. 어떻게 움직일 지 결정. getAction.
    # arbitrary depth에 대한 minimax tree.
    def value(self, gameState, depth, agentIndex) :
        
        # terminal state이면 utility 반환.
        if gameState.isWin() or gameState.isLose() : 
            return self.evaluationFunction(gameState)

        # 현재 깊이가 설정해놓은 깊이와 같다면, utility 반환.
        elif depth == self.depth : 
            return self.evaluationFunction(gameState)

        # 팩맨. 가능한 움직임 없으면 state 넣고 나온 utility 반환. 최대값으로.
        elif agentIndex == 0 :
            return self.maxV(gameState, depth)

        # 이외의 경우 다음 state로 utility 구함. 최소값으로.
        else :
            return self.minV(gameState, depth, agentIndex)

    # def max-value 구현.
    def maxV(self, gameState, depth) : 
        v = -float("inf") # 초기 값을 -infinity로 지정.

        # state의 successor 마다, successor의 값 따라서 최대값 리턴.
        for action in gameState.getLegalActions(0) : 
            v = max(v, self.value(gameState.generateSuccessor(0, action), depth, 1))

        return v

    # def min-value 구현.
    def minV(self, gameState, depth, agentIndex) : 
        v = float("inf") # 초기 값을 +infinity로 지정.

        # state의 successor 마다, successor의 값 따라서 최소값 리턴.
        # max layer에 대해 multiple min layers 대응.
        # agent index를 증가시켜가며 움직임.
        for action in gameState.getLegalActions(agentIndex) : 
            if agentIndex == gameState.getNumAgents() - 1 :
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth+1, 0))

            else : 
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1))

        return v

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # AlphaBeta pruning은 pruning algorithm.
        # minimax search와 유사하나, pruning이 존재.
        # 현재 값이 최대값보다 나은 것이 아니라면, search 멈추고 최소값 반환. pruning.
        # pruning으로 인해 더 나은 값을 놓칠 가능성 존재.

        v = -float("inf") # 초기 값을 -infinity로 지정. 최대값을 구하기 때문에.
        a = -float("inf") # alpha. 초기값 -infinity. MAX 관련.
        b = float("inf") # beta. 초기값 +infinity. MIN 관련.
        minimaxAction = Directions.STOP # 움직임을 정지 상태로 선언.

        # 가능한 움직임마다의 반복문 실행.
        for action in gameState.getLegalActions(0) : 
            newValue = self.value(gameState.generateSuccessor(0, action), 0, 1, a, b)

            # 최소값 중 최대값 구하기 위한 비교.
            if newValue > v : 
                v = newValue
                minimaxAction = action

            # pruning을 위한 alpha 값을 최대값으로 설정.
            a = max(a, v)

        return minimaxAction


    # def value 구현. 어떻게 움직일 지 결정. getAction.
    # pruning을 위한 alpha, beta 요소 추가.
    def value(self, gameState, depth, agentIndex, a, b) : 
        
        # terminal state이면 utility 반환.
        if gameState.isWin() or gameState.isLose() : 
            return self.evaluationFunction(gameState)

        # 현재 깊이가 설정해놓은 깊이와 같다면, utility 반환.
        elif depth == self.depth : 
            return self.evaluationFunction(gameState)

        # 팩맨. 가능한 움직임 없으면 state 넣고 나온 utility 반환. 최대값으로.
        elif agentIndex == 0 :
            return self.maxV(gameState, depth, a, b)

        # 이외의 경우 다음 state로 utility 구함. 최소값으로.
        else :
            return self.minV(gameState, depth, agentIndex, a, b)


    # def max-value 구현.
    def maxV(self, gameState, depth, a, b) : 
        v = -float("inf") # 초기 값을 -infinity로 지정.

        # state의 successor 마다, successor의 값 따라서 최대값 리턴.
        for action in gameState.getLegalActions(0) : 
            v = max(v, self.value(gameState.generateSuccessor(0, action), depth, 1, a, b))

            # pruning 과정.
            # 값이 beta 보다 큰지 검사. 크다면 자식 노드 검사할 필요 없으므로 반환.
            if v > b : 
                return v

            # alpha 값을 값들 중 최대값으로 설정.
            a = max(a, v)

        return v


    # def min-value 구현.
    def minV(self, gameState, depth, agentIndex, a, b) :
        v = float("inf") # 초기 값을 +infinity로 지정.

        # state의 successor 마다, successor의 값 따라서 최소값 리턴.
        # max layer에 대해 multiple min layers 대응.
        # agent index를 증가시켜가며 움직임.
        for action in gameState.getLegalActions(agentIndex) : 
            if agentIndex == gameState.getNumAgents() - 1 :
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth+1, 0, a, b))

            else : 
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, a, b))

            # pruning 과정.
            # 값이 alpha보다 작은지 검사. 작으면 자식 노드 검사할 필요 없으므로 반환.
            if v < a : 
                return v

            # beta 값을 값들 중 최소값으로 설정.
            b = min(b, v)

        return v

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
