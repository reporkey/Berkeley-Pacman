import numpy as np
class Mdp:
    def __init__(self,gameState):
        self.gameState=gameState
        self.states = np.ones((3, 4), dtype=None)
        self.states[1, 1] = -1
        walls=[[1,1],[2,2]]
        for i in walls:
            for x in range(4):
                for y in range(4):
                    if [x, y] == i:
                        self.states[3-x][y] = None
        print(self.states)
    def getStates(self):
        return self.states
class Robot(object):
    def __init__(self, Mdp):
        self.mdp = Mdp(gameState)
        states = self.mdp.getStates()
        print(states)
robo=Robot(Mdp)
print(robo.mdp.getStates())