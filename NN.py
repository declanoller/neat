import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):

	def __init__(self, D_in, H, D_out, NL_fn=torch.tanh, softmax=False):
		super(NN, self).__init__()


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





























#
