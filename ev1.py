import sys
sys.path.append('./classes')
from EPANN import EPANN
from Population import Population
import RunTools as rt
from GymAgent import GymAgent
from time import time
import numpy as np

ea = EPANN(agent_class=GymAgent, env_name='CartPole')

#ea.plotNetwork()

ea.addConnectingWeight((0,4))
ea.addConnectingWeight((1,4))
ea.addNodeInBetween(1,4)

ea.addConnectingWeight((2,5))
ea.addNode()
ea.addConnectingWeight((2,6))
ea.addConnectingWeight((3,4))
ea.addConnectingWeight((3,5))
ea.addConnectingWeight((6,5))
#ea.addAtomInBetween((2,5))


N_tests = 100000

inputs = np.random.random((N_tests, 4))

st = time()
for i in range(N_tests):

    ea.forwardPass(inputs[i])


print('time elapsed:', time() - st)

#ea.plotNetwork()




exit()


p1 = Population(agent_class=GymAgent, env_name='CartPole', N_pop=64, mut_type='change_topo', std=1.0, render_type='gym')

p1.evolve(N_gen=128, N_episode_steps=200, N_trials_per_agent=2, N_runs_with_best=2, record_final_runs=False, show_final_runs=False)

exit(0)




evolve_params = {
'N_runs' : 3,
'agent_class' : GymAgent,
'env_name' : 'LunarLander',
'N_pop' : 64,
'mut_type' : 'change_topo',
'std' : [0.01, 0.1, 1, 10],
'N_gen' : 256,
'N_trials_per_agent' : 2,
'N_runs_with_best' : 9,
'record_final_runs' : True,
'show_final_runs' : False
}


rt.varyParam(object_class=Population, run_fn=Population.evolve, run_result_var='best_individ_avg_score', **evolve_params)

exit()




evolve_params = {
'N_runs' : 3,
'agent_class' : GymAgent,
'env_name' : 'LunarLander',
'N_pop' : 64,
'mut_type' : 'change_topo',
'std' : [0.01, 0.1, 1.0, 10.0],
'N_gen' : 256,
'N_trials_per_agent' : 2,
'N_runs_with_best' : 9,
'record_final_runs' : True,
'show_final_runs' : False
}









e = EPANN(agent_class=PendulumAgent)

e.loadNetworkFromFile(
'/home/declan/Documents/code/evo1/misc_runs/evolve_22-01-2019_18-01-04__PendulumAgent' +
'/' + 'bestNN_PendulumAgent_22-01-2019_18-01-04' + '.json'
)

exit(0)







#
