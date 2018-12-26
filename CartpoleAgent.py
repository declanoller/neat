import matplotlib.pyplot as plt
import numpy as np
import gym

class CartpoleAgent:


    def __init__(self):

        self.env = gym.make('CartPole-v0')

        self.N_state_terms = len(self.env.reset())
        self.N_actions = self.env.action_space.n
        self.state = self.env.reset()



    def getStateVec(self):
        return(self.state)


    def initEpisode(self):
        self.state = self.env.reset()


    def iterate(self, action):
        # Action 0 is go L, action 1 is go R.
        observation, reward, done, info = self.env.step(action)
        self.state = observation

        return(reward, self.state, done)






    def drawState(self):

        self.env.render()







#
