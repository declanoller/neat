import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import wrappers
import FileSystemTools as fst

class LunarLanderAgent:


    def __init__(self, **kwargs):

        self.env = gym.make('LunarLander-v2')
        gym.logger.set_level(40)
        self.state_labels = ['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'v_ang']
        self.action_labels = ['nothing', 'engine_L', 'engine_main', 'engine_R',]
        # Last two states are whether the legs are touching the ground or not.
        # I'm not including them here.
        self.N_state_terms = 6
        self.N_actions = self.env.action_space.n
        self.state = self.env.reset()
        dt = fst.getDateString()
        self.base_name = f'LunarLander_{dt}'
        self.run_dir = kwargs.get('run_dir', '/home/declan/Documents/code/evo1/save_runs/')
        self.monitor_is_on = False




    def setMonitorOn(self):
        # It seems like when I call this, it gives a warning about the env not being
        # made with gym.make (which it is...), but if I call it only once for the same
        # agent, it doesn't run it every time I call it?
        #if not self.monitor_is_on:
        if True:
            self.env = wrappers.Monitor(self.env, fst.combineDirAndFile(self.run_dir, self.base_name))
            self.monitor_is_on = True


    def getStateVec(self):
        return(self.state[:self.N_state_terms])


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
