from EPANN import EPANN
from Node import Node
from Walker_1D import Walker_1D
from Population import Population
import PopTests as pt
from PuckworldAgent import PuckworldAgent
from CartpoleAgent import CartpoleAgent
from PendulumAgent import PendulumAgent
from LunarLanderAgent import LunarLanderAgent
from time import time






# add thing to make it plan N runs with the best individ, to show it actually getting better as opposed to just lucky
###### Add node legend for plotnetwork!
############## oh shit, is it gonna shit the bed when it tries to add another weight but can't?? fix that for sure
p1 = Population(agent_class=LunarLanderAgent, N_pop=128, mut_type='change_topo', std=1.0, render_type='gym', dir='misc_runs')

p1.evolve(N_gen=16, N_episode_steps=500, N_trials_per_agent=2, N_runs_with_best=9, record_final_runs=True, show_final_runs=False)

#p1.population[0].plotNetwork()
exit(0)


e = EPANN(agent_class=LunarLanderAgent, render_type='gym')
for i in range(10):
    start = time()
    e.runEpisode(400, plot_run=False)
    print('time for ep:', time()-start)


exit(0)

e = EPANN(agent_class=LunarLanderAgent, render_type='gym', N_init_hidden_nodes=0, init_IO_weights=False, verbose=True)


e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()
e.mutateAddWeight()
e.plotNetwork()








start = time()
e.clearAllNodes()
s = e.agent.getStateVec()
e.forwardPass(s)
end = time()

print('elapsed time:', end-start)

exit()




# This will be my "standard base case", that I'll center the others around, since I know it works.
'''
pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=16, N_gen=100, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)
'''

'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=[0.001, 0.01, 0.1, 1.0, 5.0], render_type='gym',
N_init_hidden_nodes=16, N_gen=150, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=[4, 8, 16, 32, 64], N_gen=150, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=[1, 5, 10], N_runs=3)


pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=5, N_runs=3, best_N_frac=[1/10.0, 1/5.0, 1/3.0])'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[8, 16, 32], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''

# Messing around with the best way to spend computation time:

'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[8, 16, 32, 64], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=[512, 256, 128, 64], N_episode_steps=400, N_trials_per_agent=5, N_runs=3)'''


'''pt.varyParam(agent_class=LunarLanderAgent, N_pop=[16, 32, 64], mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=250, N_episode_steps=400, N_trials_per_agent=[8, 4, 2], N_runs=3)'''

exit(0)

pt.varyParam(agent_class=LunarLanderAgent, N_pop=16, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=8, N_gen=[64, 128, 256, 512], N_episode_steps=400, N_trials_per_agent=[16, 8, 4, 2], N_runs=3)





exit(0)

p1 = Population(agent_class=LunarLanderAgent, N_pop=25, mut_type='gauss_noise', std=1.0, render_type='gym',
N_init_hidden_nodes=24)

p1.evolve(N_gen=50, N_steps=400, N_runs=5)
p1.population[0].plotNetwork()
p1.population[0].printNetwork()

for i in range(10):
    r = p1.population[0].runEpisode(400, plot_run=False)
    print('r of best individ: ', r)

exit()
exit()




e = EPANN(agent_class=CartpoleAgent, render_type='gym')
e.runEpisode(400, plot_run=True)
exit(0)


'''p1.population[0].printNetwork()
p1.population[0].mutate()
p1.population[0].printNetwork()
exit()'''
#ep.runEpisode(200)







'''
e = EPANN(agent_class=Walker_1D)
e.printNetwork()
e.addNode(0, 2)
e.printNetwork()
e.addNode(0, 3)
e.printNetwork()
e.addNode(5, 3)
e.printNetwork()
e.addConnection(4, 5)
e.printNetwork()
print(e.nodeInLineage(4, 6))'''







#
