from EPANN import EPANN
from Population import Population
import RunTools as rt
from GymAgent import GymAgent



p1 = Population(agent_class=GymAgent, env_name='LunarLander', N_pop=4, mut_type='change_topo', std=1.0, render_type='gym')

p1.evolve(N_gen=4, N_episode_steps=100, N_trials_per_agent=1, N_runs_with_best=3, record_final_runs=True, show_final_runs=False)

exit(0)



evolve_params = {
'N_runs' : 2,
'agent_class' : CartPoleAgent,
'N_pop' : 4,
'mut_type' : 'change_topo',
'std' : 1.0,
'N_gen' : [4, 8],
'N_episode_steps' : 100,
'N_trials_per_agent' : 1,
'N_runs_with_best' : 1,
'record_final_runs' : False,
'show_final_runs' : False
}


rt.varyParam(object_class=Population, run_fn=Population.evolve, run_result_var='best_individ_avg_score', **evolve_params)

exit()







e = EPANN(agent_class=PendulumAgent)

e.loadNetworkFromFile(
'/home/declan/Documents/code/evo1/misc_runs/evolve_22-01-2019_18-01-04__PendulumAgent' +
'/' + 'bestNN_PendulumAgent_22-01-2019_18-01-04' + '.json'
)

exit(0)







#
