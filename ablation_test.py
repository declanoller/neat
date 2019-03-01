import sys
sys.path.append('./classes')
from EPANN import EPANN
from GymAgent import GymAgent
import matplotlib.pyplot as plt
import FileSystemTools as fst
import numpy as np

params = {}
params['env_name'] = 'Pendulum'

e = EPANN(agent_class=GymAgent, env_name=params['env_name'])

path = '/home/declan/Documents/code/evo1/save_runs/evolve_23-01-2019_18-45-18__GymAgentPendulum_good'
params['NN_file'] = fst.combineDirAndFile(path, 'bestNN_GymAgent_23-01-2019_18-45-18' + '.json')

e.loadNetworkFromFile(params['NN_file'])

datetime_str = fst.getDateString()
dir = fst.combineDirAndFile('misc_runs', 'ablation_{}_{}'.format(datetime_str, params['env_name']))
fst.makeDir(dir)
plot_dir = fst.makeDir(fst.combineDirAndFile(dir, 'plots'))

log_output_str = ''

params['N_runs_per_NN'] = 50
params['N_episode_steps'] = e.agent.max_episode_steps

params['N_weights_to_remove'] = len(e.weights_list)

ablation_FF_mean_std = []

for w_removed in range(params['N_weights_to_remove']):

    # Run the ablation for several times to get stats
    ablation_scores = []
    for run in range(params['N_runs_per_NN']):
        ablation_scores.append(e.runEpisode(params['N_episode_steps']))

    # Add the mean and std
    ablation_FF_mean_std.append([w_removed, np.mean(ablation_scores), np.std(ablation_scores)])

    # Save what the NN currently looks like
    NN_save_fname = fst.combineDirAndFile(plot_dir, 'NN_plot_{}w_removed.png'.format(w_removed))
    e.plotNetwork(show_plot=False, save_plot=True, fname=NN_save_fname, node_legend=True)

    # Remove the next smallest weight
    smallest_weight_connection = min(e.weights_dict, key=lambda x: abs(e.weights_dict.get(x)))
    remove_str = 'Removing weight {} that has value {:.3f}\n'.format(smallest_weight_connection, e.weights_dict[smallest_weight_connection])
    print(remove_str)
    log_output_str += remove_str
    e.removeConnectingWeight(smallest_weight_connection)



# Save params
fst.writeDictToFile(params, fst.combineDirAndFile(dir, 'Params_logfile_{}.log'.format(datetime_str)))

# Save weight order removal
removal_log_fname = fst.combineDirAndFile(dir, 'Weight_remove_order_{}.txt'.format(datetime_str))
with open(removal_log_fname, 'w+') as f:
    f.write(log_output_str)

# Plot the mean and std FF as a function of removing weights
ablation_FF_mean_std = np.array(ablation_FF_mean_std)
weights_removed = ablation_FF_mean_std[:, 0]
FF_mean = ablation_FF_mean_std[:, 1]
FF_std = ablation_FF_mean_std[:, 2]
plt.fill_between(
np.array(range(len(FF_mean))),
FF_mean - FF_std,
FF_mean + FF_std,
facecolor='dodgerblue', alpha=0.5)

plt.plot(FF_mean, color='mediumblue')
plt.xlabel('# weights removed')
plt.ylabel('FF')
plt.title('Ablation test, FF over {} episodes each'.format(params['N_runs_per_NN']))
fname = fst.combineDirAndFile(dir, '{}_{}.png'.format('ablation_FF_mean-std_plot', datetime_str))
plt.savefig(fname)

# Save mean/std
fname = fst.combineDirAndFile(dir, '{}_{}.txt'.format('ablation_FF_mean-std', datetime_str))
np.savetxt(fname, ablation_FF_mean_std, fmt='%.4f')




#
