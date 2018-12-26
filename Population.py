from EPANN import EPANN
from copy import deepcopy
import matplotlib.pyplot as plt
import FileSystemTools as fst
from statistics import mean, stdev
import PopTests as pt
import numpy as np

class Population:


    def __init__(self, agent_class, **kwargs):

        self.N_pop = kwargs.get('N_pop', 15)
        self.agent_class = agent_class
        self.mut_type = kwargs.get('mut_type', 'change_topo')
        self.gauss_std = kwargs.get('std', 0.2)
        self.best_N_frac = kwargs.get('best_N_frac', 1/5.0)

        self.fname_notes = kwargs.get('fname_notes', '')
        self.dir = fst.makeLabelDateDir('evolve_{}_'.format(self.fname_notes), base_dir='misc_runs')
        self.plot_dir = fst.makeDir(fst.combineDirAndFile(self.dir, 'plots'))
        print('run dir: ', self.dir)
        pop_kwargs = {'agent_class' : self.agent_class, 'run_dir' : self.dir}
        both_kwargs = {**kwargs, **pop_kwargs}

        self.population = [EPANN(**both_kwargs) for i in range(self.N_pop)]



    def evolve(self, **kwargs):


        start_time = fst.getCurTimeObj()

        N_trials_per_agent = kwargs.get('N_trials_per_agent', 3)
        N_episode_steps = kwargs.get('N_episode_steps', 400)
        N_gen = kwargs.get('N_gen', 50)
        N_trials_per_agent = kwargs.get('N_trials_per_agent', 3)
        N_runs_each_champion = kwargs.get('N_runs_each_champion', 10)
        N_runs_with_best = kwargs.get('N_runs_with_best', 10)
        record_final_runs = kwargs.get('record_final_runs', False)
        show_final_runs = kwargs.get('show_final_runs', False)

        best_FFs = []
        mean_FFs = []

        all_FFs = []
        all_nodecounts = []
        all_weightcounts = []
        champion_FF_mean_std = [] # A list of pairs of [mean, std] for runs of the current champion.


        #try:
        for i in range(N_gen):

            best_FF = -100000000
            mean_FF = 0

            mean_Rs = []
            for j, individ in enumerate(self.population):
                mean_episode_score = 0
                for run in range(N_trials_per_agent):
                    mean_episode_score += individ.runEpisode(N_episode_steps)

                mean_episode_score = mean_episode_score/N_trials_per_agent
                mean_Rs.append([j, mean_episode_score])
                mean_FF += mean_episode_score
                if mean_episode_score > best_FF:
                    best_FF = mean_episode_score

            mean_FF = mean_FF/self.N_pop
            best_FFs.append(best_FF)
            mean_FFs.append(mean_FF)

            mean_Rs_no_label = [x[1] for x in mean_Rs]

            if i%max(1, int(N_gen/20))==0:
                print('\ngen {:.1f}. Best FF = {:.4f}, mean FF = {:.4f}'.format(i, best_FF, mean_FF))
                self.plotPopHist(mean_Rs_no_label, 'pop_FF')
                if self.mut_type == 'change_topo':
                    self.plotPopHist([len(epann.node_list) for epann in self.population], 'pop_nodecount')
                    self.plotPopHist([len(epann.weights_list) for epann in self.population], 'pop_weightcount')
                    fname = fst.combineDirAndFile(self.plot_dir, '{}_{}.png'.format('bestNN', fst.getDateString()))
                    self.population[0].plotNetwork(show_plot=False, save_plot=True, fname=fname, node_legend=True)

                    print('network sizes: ', [len(x.node_list) for x in self.population])
                    print('avg network size: {:.3f}'.format(sum([len(x.node_list) for x in self.population])/self.N_pop))
                    print('# network connections: ', [len(x.weights_list) for x in self.population])
                    print('avg # network cennections: {:.3f}'.format(sum([len(x.weights_list) for x in self.population])/self.N_pop))

            all_FFs.append(mean_Rs_no_label)
            all_nodecounts.append([len(epann.node_list) for epann in self.population])
            all_weightcounts.append([len(epann.weights_list) for epann in self.population])

            self.getNextGen(mean_Rs)

            champion_scores = []
            for run in range(N_runs_each_champion):
                champion_scores.append(self.population[0].runEpisode(N_episode_steps))

            champion_FF_mean_std.append([np.mean(champion_scores), np.std(champion_scores)])

        #except:
        #    print('Evo. canceled, trying to finish evolve() function.')

        final_sorted = self.sortByFitnessFunction(mean_Rs)
        print(final_sorted)

        print('\n\nRun took: ', fst.getTimeDiffStr(start_time), '\n\n')

        self.saveScore(best_FFs, 'bestscore')
        self.saveScore(mean_FFs, 'meanscore')
        self.saveScore(all_FFs, 'all_FFs')
        self.saveScore(all_nodecounts, 'nodecounts')
        self.saveScore(all_weightcounts, 'weightcounts')
        self.saveScore(champion_FF_mean_std, 'champion_FF_mean_std')

        # Plot best and mean FF curves for the population
        plt.subplots(1, 1, figsize=(8,8))
        plt.plot(mean_FFs, color='dodgerblue', label='Pop. avg FF')
        plt.plot(best_FFs, color='tomato', label='Pop. best FF')
        plt.xlabel('generations')
        plt.ylabel('FF')
        plt.legend()
        fname = fst.combineDirAndFile(self.dir, '{}_{}.png'.format('FFplot', fst.getDateString()))
        plt.savefig(fname)

        plt.close()

        # Plot the mean and std for the champion of each generation
        champion_FF_mean_std = np.array(champion_FF_mean_std)
        champ_mean = champion_FF_mean_std[:,0]
        champ_std = champion_FF_mean_std[:,1]
        plt.fill_between(np.array(range(len(champ_mean))), champ_mean - champ_std, champ_mean + champ_std, facecolor='dodgerblue', alpha=0.5)
        plt.plot(champ_mean, color='mediumblue')
        plt.xlabel('generations')
        plt.ylabel('FF')
        fname = fst.combineDirAndFile(self.dir, '{}_{}.png'.format('champion_mean-std_plot', fst.getDateString()))
        plt.savefig(fname)

        # Get an avg final score for the best individ.
        best_individ = self.population[0]
        if record_final_runs:
            best_individ_scores = [best_individ.runEpisode(N_episode_steps, plot_run=show_final_runs, **kwargs, record_episode=True) for i in range(N_runs_with_best)]
        else:
            best_individ_scores = [best_individ.runEpisode(N_episode_steps, plot_run=show_final_runs, **kwargs) for i in range(N_runs_with_best)]

        best_individ_avg_score = mean(best_individ_scores)

        # Plot some more stuff with the saved dat
        pt.plotPopulationProperty(self.dir, 'all_FFs')
        pt.plotPopulationProperty(self.dir, 'weightcounts')

        return(best_FFs, mean_FFs, best_individ_avg_score)


    def getNextGen(self, FF_list):

        pop_indices_sorted = self.sortByFitnessFunction(FF_list)
        best_N = max(int(self.N_pop*self.best_N_frac), 2)
        #best_N = 1
        top_N_indices = [x[0] for x in pop_indices_sorted[:best_N]]
        #print('best individ:')
        #self.population[top_N_indices[0]].printNetwork()



        new_pop = [self.population[top_N_indices[0]].clone()]
        mod_counter = 0

        while len(new_pop)<self.N_pop:

            new_EPANN = self.population[top_N_indices[mod_counter%best_N]].clone()
            if self.mut_type == 'change_topo':
                new_EPANN.mutate(std=self.gauss_std)
            if self.mut_type == 'gauss_noise':
                new_EPANN.gaussMutate(std=self.gauss_std)

            new_pop.append(new_EPANN)
            mod_counter += 1

        self.population = new_pop


    def saveScore(self, score, label):
        fname = fst.combineDirAndFile(self.dir, '{}_{}.txt'.format(label, fst.getDateString()))
        np.savetxt(fname, score, fmt='%.4f')


    def sortByFitnessFunction(self, FF_list):

        # Assumes self.pop is in the same order it was when it was evaluated.

        pop_indices_sorted = sorted(FF_list, key=lambda x: -x[1])
        return(pop_indices_sorted)


    def plotPopHist(self, hist_var, label):
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.hist(hist_var, facecolor='dodgerblue', edgecolor='k', label=label)
        plt.axvline(np.mean(hist_var), color='k', linestyle='dashed', linewidth=1)
        plt.xlabel(label)
        ax.legend()
        fname = fst.combineDirAndFile(self.plot_dir, '{}_{}.png'.format(label, fst.getDateString()))
        plt.savefig(fname)

        # Is this the way to make sure too many figures don't get created? or plt.close()?
        plt.close()





#
