from EPANN import EPANN
from Node import Node
from Walker_1D import Walker_1D
from Population import Population
import PopTests as pt
import RunTools as rt
from PuckworldAgent import PuckworldAgent
from CartPoleAgent import CartPoleAgent
from PendulumAgent import PendulumAgent
from LunarLanderAgent import LunarLanderAgent
from time import time



brach_params = {
'individ_class' : Brachistochrone,
'N' : 5,
'height' : 1.3,
'init_T_factor' : 0.00001,
'mutate_strength_height_frac' : [0.05, 0.5],
'N_gen' : 20,
'make_gif' : False,
'show_plot' : False,
'N_runs' : 2,
'save_FF' : True
}



rt.varyParam(object_class=Population, run_fn=Population.evolve, run_result_var='best_FF', **brach_params)



exit()


p1 = Population(agent_class=PendulumAgent, N_pop=4, mut_type='change_topo', std=1.0, render_type='gym')

p1.evolve(N_gen=8, N_episode_steps=100, N_trials_per_agent=1, N_runs_with_best=1, record_final_runs=False, show_final_runs=False)

exit(0)




e = EPANN(agent_class=PendulumAgent)

e.loadNetworkFromFile(
'/home/declan/Documents/code/evo1/misc_runs/evolve_22-01-2019_18-01-04__PendulumAgent' +
'/' + 'bestNN_PendulumAgent_22-01-2019_18-01-04' + '.json'
)

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
