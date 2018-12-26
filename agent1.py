import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from copy import deepcopy


class DQN(nn.Module):

	def __init__(self, D_in, H, D_out, NL_fn=torch.tanh, softmax=False):
		super(DQN, self).__init__()

		self.lin1 = nn.Linear(D_in,H)
		self.lin2 = nn.Linear(H,D_out)
		self.NL_fn = NL_fn
		self.softmax = softmax

	def forward(self, x):
		x = self.lin1(x)
		#x = F.relu(x)
		#x = torch.tanh(x)
		x = self.NL_fn(x)
		x = self.lin2(x)
		if self.softmax:
			x = torch.softmax(x,dim=1)
		return(x)



class agent1:


	def __init__(self, **kwargs):

		self.xlims = kwargs.get('xlims', np.array([-0.5,0.5]))
		self.ylims = kwargs.get('ylims', np.array([-0.5,0.5]))
		self.lims = np.array((self.xlims,self.ylims))
		self.max_dist = sqrt(np.ptp(self.xlims)**2 + np.ptp(self.ylims)**2)
		self.a = kwargs.get('a',1.0)
		self.drag = kwargs.get('drag', 0.5)
		self.time_step = kwargs.get('dt',10**-1)
		self.reward_type = kwargs.get('reward','sparse')

		self.passed_params = {}
		check_params = ['a', 'drag', 'dt', 'reward']
		for param in check_params:
			if kwargs.get(param, None) is not None:
				self.passed_params[param] = kwargs.get(param, None)

		self.N_actions = 4


		self.circ_rad = np.ptp(self.xlims)/20.0
		self.target_rad = 1*self.circ_rad
		self.resetTarget()

		self.pos0 = np.array([self.xlims.mean()/2.0,self.ylims.mean()/2.0])
		self.v0 = np.array([0.0,0.0])
		self.resetStateValues()
		self.accel_array = np.array([[0,1],[0,-1],[-1,0],[1,0]])



		self.N_state_terms = len(self.getStateVec())

		self.HLN = 20

		self.dtype = torch.float32
		torch.set_default_dtype(self.dtype)

		# I think it's already randomly initializing the weights with a gaussian mean=0, std=1
		self.policy_NN = DQN(self.N_state_terms, self.HLN, self.N_actions, softmax=True)

		'''for p in self.policy_NN.parameters():
			print(p.data)'''
		self.N_weight_tensors = len(list(self.policy_NN.parameters()))


		self.N_mate_swaps = 18
		self.N_mutations = 2



	def puckTargetDist(self):
		return(sqrt(np.sum((self.pos-self.target)**2)))


	def addToHist(self):
		self.pos_hist = np.concatenate((self.pos_hist,[self.pos]))
		self.v_hist = np.concatenate((self.v_hist,[self.v]))
		self.t.append(self.t[-1] + self.time_step)
		self.r_hist.append(self.reward())


	def resetTarget(self):

		self.target = self.target_rad + self.lims[:,0] + np.random.random((2,))*(np.ptp(self.lims,axis=1)-2*self.target_rad)


	def iterateEuler(self,action):

		#this uses the Euler-Cromer method to move.

		#Right now I'm just gonna make it sit against a wall if it goes to the
		#boundary, but it might be cool to make periodic bry conds, to see if it would
		#learn to zoom around it.

		a = self.actionToAccel(action) - self.drag*self.v

		v_next = self.v + a*self.time_step
		pos_next = self.pos + v_next*self.time_step

		#To handle the walls
		for i in [0,1]:
			if pos_next[i] < (self.lims[i,0] + self.circ_rad):
				pos_next[i] = self.lims[i,0] + self.circ_rad
				# This makes it "bounce" off the wall, so it keeps momentum.
				v_next[i] = -v_next[i]
				# This makes it "stick" to the wall.
				#v_next[i] = 0

			if pos_next[i] > (self.lims[i,1] - self.circ_rad):
				pos_next[i] = self.lims[i,1] - self.circ_rad
				v_next[i] = -v_next[i]
				#v_next[i] = 0

		self.pos = pos_next
		self.v = v_next
		self.addToHist()


	def actionToAccel(self,action):
		self.a_hist.append(action)
		return(self.a*self.accel_array[action])


	def softmaxAction(self, state_vec):
		pi_vals = self.policy_NN(state_vec)
		m = Categorical(pi_vals)
		return(m.sample())

	###################### Required agent functions


	def mate(self, other_agent):

		ag1 = deepcopy(self)
		ag2 = deepcopy(other_agent)

		lin1_weight_shape = ag1.policy_NN.lin1.weight.data.shape
		lin1_bias_shape = ag1.policy_NN.lin1.bias.data.shape
		lin2_weight_shape = ag1.policy_NN.lin2.weight.data.shape
		lin2_bias_shape = ag1.policy_NN.lin2.bias.data.shape


		for i in range(self.N_mate_swaps):

			r1 = np.random.randint(0, lin1_weight_shape[0])
			r2 = np.random.randint(0, lin1_weight_shape[1])
			ag1.policy_NN.lin1.weight.data[r1,r2], ag2.policy_NN.lin1.weight.data[r1,r2] = ag2.policy_NN.lin1.weight.data[r1,r2], ag1.policy_NN.lin1.weight.data[r1,r2]

			r1 = np.random.randint(0, lin2_weight_shape[0])
			r2 = np.random.randint(0, lin2_weight_shape[1])
			ag1.policy_NN.lin2.weight.data[r1,r2], ag2.policy_NN.lin2.weight.data[r1,r2] = ag2.policy_NN.lin2.weight.data[r1,r2], ag1.policy_NN.lin2.weight.data[r1,r2]

			r1 = np.random.randint(0, lin1_weight_shape[0])
			ag1.policy_NN.lin1.bias.data[r1], ag2.policy_NN.lin1.bias.data[r1] = ag2.policy_NN.lin1.bias.data[r1], ag1.policy_NN.lin1.bias.data[r1]

			r1 = np.random.randint(0, lin2_weight_shape[0])
			ag1.policy_NN.lin2.bias.data[r1], ag2.policy_NN.lin2.bias.data[r1] = ag2.policy_NN.lin2.bias.data[r1], ag1.policy_NN.lin2.bias.data[r1]

		return(ag1, ag2)


	def isSameState(self, other_agent):
		return(False)


	def mutate(self):

		lin1_weight_shape = self.policy_NN.lin1.weight.data.shape
		lin1_bias_shape = self.policy_NN.lin1.bias.data.shape
		lin2_weight_shape = self.policy_NN.lin2.weight.data.shape
		lin2_bias_shape = self.policy_NN.lin2.bias.data.shape

		for i in range(self.N_mutations):

			r1 = np.random.randint(0, lin1_weight_shape[0])
			r2 = np.random.randint(0, lin1_weight_shape[1])
			self.policy_NN.lin1.weight.data[r1,r2] = np.random.randn()

			r1 = np.random.randint(0, lin2_weight_shape[0])
			r2 = np.random.randint(0, lin2_weight_shape[1])
			self.policy_NN.lin2.weight.data[r1,r2] = np.random.randn()

			r1 = np.random.randint(0, lin1_weight_shape[0])
			self.policy_NN.lin1.bias.data[r1] = np.random.randn()

			r1 = np.random.randint(0, lin2_weight_shape[0])
			self.policy_NN.lin2.bias.data[r1] = np.random.randn()



	def fitnessFunction(self):
		self.fixedLengthEpisode(100)
		# I think the fitness function is meant to be minimized, so we should pass it
		# the negative of the total reward.
		return(-sum(self.r_hist))



	def fixedLengthEpisode(self, N_steps):
		self.resetTarget()
		self.resetStateValues()

		for i in range(N_steps):
		    s = torch.tensor(self.getStateVec(), dtype=torch.float32).unsqueeze(dim=0)
		    a = self.softmaxAction(s)
		    r, s_next = self.iterate(a)


	def getPassedParams(self):
		#This returns a dict of params that were passed to the agent, that apply to the agent.
		#So if you pass it a param for 'reward', it will return that, but it won't return the
		#default val if you didn't pass it.
		return(self.passed_params)


	def getStateVec(self):
		assert self.target is not None, 'Need target to get state vec'
		return(np.concatenate((self.pos,self.v,self.target)))


	def getState(self):
		return(self.getStateVec())

	def printState(self):
		print(self.getState())

	def reward(self):

		assert self.target is not None, 'Need a target'

		max_R = 1

		if self.reward_type == 'sparse':
			if self.puckTargetDist() <= (self.target_rad + self.circ_rad):
				return(max_R)
			else:
				return(-0.01)

		if self.reward_type == 'shaped':
			#return(max_R*(self.max_dist/2.0 - self.puckTargetDist()))
			#These numbers will probably have to change if a, dt, or the dimensions change.
			return(-0.5*self.puckTargetDist() + 0.4)


	def initEpisode(self):
		self.resetStateValues()
		self.resetTarget()


	def iterate(self,action):
		self.iterateEuler(action)

		r = self.reward()
		if r > 0:
			self.resetTarget()

		return(r,self.getStateVec())


	def resetStateValues(self):

		self.pos = self.pos0
		self.v = self.v0

		self.pos_hist = np.array([self.pos])
		self.v_hist = np.array([self.v])
		self.t = [0]
		self.a_hist = [0]
		self.r_hist = []


	def drawState(self, ax):

		ax.clear()
		ax.set_xlim(tuple(self.xlims))
		ax.set_ylim(tuple(self.ylims))

		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_aspect('equal')

		puck = plt.Circle(tuple(self.pos), self.circ_rad, color='tomato')
		ax.add_artist(puck)

		if self.target is not None:
			target = plt.Circle(tuple(self.target), self.target_rad, color='seagreen')
			ax.add_artist(target)


	def plotStateParams(self,axes):

		ax1 = axes[0]
		ax2 = axes[1]
		ax3 = axes[2]
		ax4 = axes[3]

		ax1.clear()
		ax1.plot(self.pos_hist[:,0][-1000:],label='x')
		ax1.plot(self.pos_hist[:,1][-1000:],label='y')
		ax1.legend()

		ax2.clear()
		ax2.plot(self.a_hist[-1000:],label='a')
		ax2.set_yticks([0,1,2,3])
		ax2.set_yticklabels(['U','D','L','R'])
		ax2.legend()


		ax3.clear()
		ax3.plot(self.r_hist[-1000:],label='R')
		ax3.legend()


		ax4.clear()
		ax4.plot(self.v_hist[:,0][-1000:],label='vx')
		ax4.plot(self.v_hist[:,1][-1000:],label='vy')
		ax4.legend()




#
