import numpy as np
import gym
from gym import wrappers
import FileSystemTools as fst
import json


'''

This is a generalized agent for OpenAI gym environments.
I'm doing it because I had to create diff. agents for each environment,
when they really just need a few specific things for each. So now you just pass it
env_name in the kwargs, and it will look up in a json file the right stuff.

Here, the env_name you pass it will be something like 'Pendulum', not Pendulum-v0,
because I don't want to have to deal with remembering versions.

See createEnvJson.py and loadEnvJson() for details.

need to provide:

--state labels (for each state var)
--action labels (for each action var)
--gym_env_name
--action_space_type

functions:

--getStateVec()
--initEpisode()
--iterate() (returns a tuple of (reward, state, boolean isDone))
--setMaxEpisodeSteps()

'''



class GymAgent:


    def __init__(self, **kwargs):

        self.env_name = kwargs.get('env_name', None)
        assert self.env_name is not None, 'Need to provide an env_name argument!'

        # Load all the properties for this env.
        self.loadEnvJson(self.env_name)

        # Create the env
        self.env = gym.make(self.gym_env_name)
        gym.logger.set_level(40)

        self.state = self.env.reset() # Should I be doing this here? sometimes trouble with resetting when done=False
        dt = fst.getDateString()
        self.base_name = f'{self.env_name}_{dt}'
        self.run_dir = kwargs.get('run_dir', '/home/declan/Documents/code/evo1/misc_runs/')
        self.monitor_is_on = False


    def setMaxEpisodeSteps(self, N_steps):

        self.env._max_episode_steps = N_steps
        self.env.spec.max_episode_steps = N_steps
        self.env.spec.timestep_limit = N_steps


    def closeEnv(self):
        # This doesn't seem to be a good idea to use with monitor?
        self.env.close()


    def setMonitorOn(self, show_run=True):
        # It seems like when I call this, it gives a warning about the env not being
        # made with gym.make (which it is...), but if I call it only once for the same
        # agent, it doesn't run it every time I call it?
        #if not self.monitor_is_on:
        #
        # Also, it seems like you can't record the episode without showing it on the screen.
        # See https://github.com/openai/gym/issues/347 maybe?

        self.record_dir = fst.combineDirAndFile(self.run_dir, self.base_name)
        #if show_run:
        if True:
            self.env = wrappers.Monitor(self.env, self.record_dir)
        else:
            self.env = wrappers.Monitor(self.env, self.record_dir, video_callable=False, force=True)
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


    def loadEnvJson(self, env_name):

        with open('gym_env_info.json') as json_file:
            env_info_dict = json.load(json_file)

        env_info = env_info_dict[env_name]

        self.gym_env_name = env_info['gym_env_name']
        self.state_labels = env_info['state_labels']
        self.action_labels = env_info['action_labels']
        self.action_space_type = env_info['action_space_type']
        self.N_state_terms = len(self.state_labels)
        self.N_actions = len(self.action_labels)







#
