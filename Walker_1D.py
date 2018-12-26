import matplotlib.pyplot as plt
import numpy as np


class Walker_1D:


    def __init__(self):

        self.lims = np.array([-1.0, 1.0])
        self.width = self.lims[1] - self.lims[0]

        self.step_size = self.width/50.0

        self.position = 0
        self.target_position = None

        self.N_state_terms = len(self.getStateVec())
        self.N_actions = 2



    def getStateVec(self):
        return(np.array([self.position, self.target_position]))


    def initEpisode(self):
        self.resetPosition()
        self.resetTarget()


    def resetTarget(self):

        x = np.random.random()
        self.target_position = self.lims[0] + self.width*x
        # print('new target pos: {:.3f}'.format(self.target_position))


    def resetPosition(self):
        self.position = 0


    def iterate(self, action):
        # Action 0 is go L, action 1 is go R.
        add_x = (action - 0.5)*2
        # maps 0,1 to -1,1
        self.position += add_x*self.step_size
        self.position = max(self.position, self.lims[0] + self.step_size)
        self.position = min(self.position, self.lims[1] - self.step_size)
        return(self.reward(), self.getStateVec(), False)



    def reward(self):

        if abs(self.position - self.target_position) <= 1.2*self.step_size:
            self.resetTarget()
            return(1.0)
        else:
            return(-0.01)



    def drawState(self, ax):

        ax.clear()
        ax.set_xlim(tuple(self.lims))
        ax.set_ylim(tuple(self.lims))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        ag = plt.Circle((self.position, 0), 0.03, color='tomato')
        ax.add_artist(ag)

        if self.target_position is not None:
            target = plt.Circle((self.target_position, 0), 0.03, color='seagreen')
            ax.add_artist(target)







#
