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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyOffense', second = 'MyDefense'):
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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

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
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


# baseline의 reflex agent를 응용한 baseline1.
class MyAgent_1(CaptureAgent):

  def registerInitialState(self, gameState):
    
    CaptureAgent.registerInitialState(self, gameState)
    self.init = gameState.getAgentPosition(self.index)

    # 쓸만한 요소들 추가.
    # 벽에 대한 정보, 도주를 위한 정보, defense를 위한 정보 등.
    self.wall = gameState.getWalls()
    self.chased = False
    self.shield = True
    self.foodNum = 0
    self.reverseS = 0

    # team에 대한 정보.
    if self.red:
      self.team = "Red"

    else:
      self.team = "Blue"

    # 레이아웃에 대한 정보. 
    layoutWidth = (gameState.data.layout.width)
    layoutHeight = (gameState.data.layout.height)
    
    if self.team == "Red":
      self.x = (layoutWidth-1)//2
    
    else:
      self.x = (layoutWidth)//2

    self.y = (layoutHeight)//2

    # 집으로 복귀를 위한 정보.
    if not self.wall[self.x][self.y]:
      self.exit = [(self.x, self.y) for y in range(1, (layoutHeight-1))]
    
  def getFeatures(self, gameState, action):

    feature = util.Counter()
    feature["succScore"] = self.getScore(gameState)

    return feature

  def getWeights(self, gameState, action):
    
    return {"succScore" : 1.0}

  def evaluate(self, gameState, action):

    feat = self.getFeatures(gameState, action)
    weight = self.getWeights(gameState, action)

    return (feat*weight)

  # 행동 방식 결정.
  def chooseAction(self, gameState):
    
    # 점수 평가 후 제일 큰 점수를 주는 방식을 선택.
    action = gameState.getLegalActions(self.index)
    value = [self.evaluate(gameState, act) for act in action]
    maxV = max(value)
    bestAct = [act for act, v in zip(action, value) if v == maxV]

    # reflex의 방식 응용. 남은 음식에 따라 행동 정함.
    leftover = len(self.getFood(gameState).asList())

    if leftover <= 2:
      bestDist = 9999

      for a in action:
        succ = gameState.generateSuccessor(self.index, a)
        curr2 = succ.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.init, curr2)

        if dist < bestDist:
          bestAct = a
          bestDist = dist

      return bestAct

    # 최적의 다음 행동 결정.
    nextAct = random.choice(bestAct)
    nextS = gameState.generateSuccessor(self.index, nextAct)
    nextCurr = nextS.getAgentPosition(self.index)

    if (self.team == "Red" and self.x < nextCurr[0]):
      self.reverseS = 0

    if (self.team == "Blue" and (self.x-1) > nextCurr[0]):
      self.reverseS = 0

    else:
      self.reverseS += 1

    # 다음 상태가 pacman이냐에 따라 초기화.
    if nextS.getAgentState(self.index).isPacman:
      self.shield = False
      self.foodNum += len(self.getFood(gameState).asList()) - len(self.getFood(nextS).asList())

    else:
      self.shield = True
      self.chased = False
      self.foodNum = 0

    return nextAct


class MyOffense(MyAgent_1):

  def getFeatures(self, gameState, action):
    
    feat = util.Counter()

    # 다음 상태에 대한 정보.
    nextS = gameState.generateSuccessor(self.index, action)
    nextCurr = nextS.getAgentPosition(self.index)

    # food와 capsule에 대한 정보.
    currFood = self.getFood(gameState).asList()
    nextFood = self.getFood(nextS).asList()
    foodDist = [self.getMazeDistance(nextCurr, f) for f in nextFood]
    currCap = self.getCapsules(gameState)
    nextCap = self.getCapsules(nextS)

    # opponent에 대한 정보. 침입자인지 유령인지.
    nextOppo = [gameState.getAgentState(o) for o in self.getOpponents(nextS)]
    nextEnemy = [p for p in nextOppo if p.isPacman and p.getPosition() != None]
    nextGhost = [g for g in nextOppo if not g.isPacman and g.getPosition() != None]
    nextSGhost = [g for g in nextOppo if not g.isPacman and g.getPosition() != None and g.scaredTimer > 0]

    # food에 대한 점수. 가까운 food를 찾아가고, 남은 음식에 따라 점수 변경.
    if not self.chased:
      feat["succScore"] = len(currFood) - len(nextFood)

    else:
      feat["succScore"] = 0

    feat["minFoodDist"] = min(foodDist)

    # 유령에 대한 점수. 유령으로부터 도망가야 성공적.
    nextGDist = [self.getMazeDistance(nextCurr, g.getPosition()) for g in nextGhost]
    threat = [self.getMazeDistance(nextCurr, g.getPosition()) for g in nextGhost if self.getMazeDistance(nextCurr, g.getPosition()) < 4]

    if not self.shield:

      if len(threat) <= 0:
        feat["ghostDist"] = 5
        self.chased = False
      
      else:
        feat["ghostDist"] = min(threat)
        self.chased = True

    else:

      if len(threat) <= 0:
        feat["ghostDist"] = 5

      else:
        feat["ghostDist"] = min(threat)

    # 점수 획득 후 집으로 복귀.
    comeback = [self.getMazeDistance(h, nextCurr) for h in self.exit]

    # 얼마나 먹었느냐에 따라 행동 결정. 많이 먹었으면 도망가기. 쫓기면 도망가기.
    if self.chased and self.foodNum == 0:
      feat["comeback"] = 0

    elif not self.chased:

      if self.foodNum >= 5:
        feat["comeback"] = (5*min(comeback))

      elif self.foodNum >= 3 and len(nextGDist) > 0 and min(nextGDist) < 6:
        feat["comeback"] = (5*min(comeback))

      elif self.foodNum == 0:
        feat["comeback"] = 0

    else:
      feat["comeback"] = min(comeback)

    # Opponent에 대한 점수. 죽일 수 있으면 죽이기.
    feat["supportDefense"] = 1

    if self.shield:
      feat["supportDefense"] = -len(nextEnemy)

    feat["killScared"] = 2

    if len(nextSGhost) > 0:
      self.chased = False
      feat["killScared"] = -len(nextSGhost)

    # Capsule에 대한 정보. 점수에 따라 캡슐 먹기.
    nextCDist = [self.getMazeDistance(nextCurr, c) for c in nextCap]

    if len(nextCDist) <= 0:
      feat["eatCap"] = (len(nextCap) - len(currCap))*100

    else:
      feat["eatCap"] = min(nextCDist)*(0.1)

    # 사망에 대한 정보.
    if self.getMazeDistance(nextCurr, self.init) >= 3:
      feat["dead"] = 0
    
    else:
      feat["dead"] = float("inf")

    # 정지 및 회귀에 대한 정보. 되도록 하지 않도록.
    if action == Directions.STOP:
      feat["stop"] = 1

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

    if action == rev:
      feat["reverse"] = 1

    return feat

  # 점수 체계. 수치는 임의로 정함.
  def getWeights(self, gameState, action):
    
    weight = {
      "succScore" : 100,
      "minFoodDist" : -2,
      "ghostDist" : 100, 
      "comeback" : -3,
      "supportDefense" : 1000,
      "killScared" : -1.5, 
      "eatCap" : -10,
      "dead" : -1,
      "stop" : -1000,
      "reverse" : -5
    }

    return weight


class MyDefense(MyAgent_1):

  def getFeatures(self, gameState, action):
    
    feat = util.Counter()

    # successor 및 다음 상태에 대한 정보.
    succ = gameState.generateSuccessor(self.index, action)
    nextS = succ.getAgentState(self.index)
    nextCurr = nextS.getPosition()

    # 음식에 대한 정보.
    nextFood = self.getFood(succ).asList()
    foods = sorted(nextFood, key = lambda x: self.getMazeDistance(self.exit[0], x))

    if len(foods) > 0:
      feat["minFoodDist"] = -self.getMazeDistance(nextCurr, foods[0])

    # 방어 상태에 대한 정보. 팩맨이면 방어 안한다. 상대 진영에서 공격.
    feat["onDefense"] = 1

    if nextS.isPacman:
      feat["onDefense"] = 0

    # opponent에 대한 정보. 위치와 거리.
    oppo = [succ.getAgentState(o) for o in self.getOpponents(succ)]
    oppoCurr = [o.getPosition() for o in oppo]
    oppoDist = [self.getMazeDistance(nextCurr, o) for o in oppoCurr]
    enemy = [e for e in oppo if e.isPacman and e.getPosition() != None]
    
    # 점수와 공수 전환. 적이 없으면 공격으로 변경. 
    feat["succScore"] = -len(nextFood)
    feat["doOffense"] = 0

    if len(enemy) == 0 and min(oppoDist) > 0:
      feat["doOffense"] = 2
      feat["invaderDist"] = 0

    # 집으로 복귀하기 위한 정보. 먹고 돌아오기. 적과의 거리에 따라서.
    comeback = [self.getMazeDistance(h, nextCurr) for h in self.exit]

    if self.foodNum > 1:
      feat["comeback"] = min(comeback)

    feat["invaderNum"] = len(enemy)

    if len(enemy) > 0:
      enemyDist = [self.getMazeDistance(nextCurr, e.getPosition()) for e in enemy]
      feat["invaderDist"] = min(enemyDist)
      feat["comeback"] = min(comeback)

    # 정지 및 회귀에 대한 정보. 되도록 하지 않도록.
    if action == Directions.STOP:
      feat["stop"] = 1

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

    if action == rev:
      feat["reverse"] = 1

    return feat

  # 점수 체계. 수치는 임의로 정함.
  def getWeights(self, gameState, action):
    
    return {
      "succScore" : 100,
      "minFoodDist" : 2,
      "invaderDist" : -100,
      "invaderNum" : -1000, 
      "comeback" : -3,
      "onDefense" : 1000,   
      "doOffense" : 1000, 
      "stop" : -1000, 
      "reverse" : -5
    }
