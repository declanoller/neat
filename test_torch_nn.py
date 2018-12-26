
from LunarLanderAgent import LunarLanderAgent
from NN import NN
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from time import time



nn1 = NN(7, 8, 4)


start = time()



end = time()

print('elapsed time:', end-start)

exit()

a = torch.tensor([6.3,7,8,2,0,5,1])

nn1.forward(a)



print(nn1.state_dict())
print('\n\n')


for k in nn1.state_dict().keys():
    print(k)
    print(nn1.state_dict()[k])
    nn1.state_dict()[k].data[0] = 100

print('\n\n')
print(nn1.state_dict())


a = torch.tensor([6,7,8,2]).float()
nn1.forward(a)













#
