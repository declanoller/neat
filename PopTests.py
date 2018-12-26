import Population
import matplotlib.pyplot as plt
from statistics import mean,stdev
import FileSystemTools as fst
from time import time
import numpy as np
import os
import glob
from math import sqrt, ceil
import subprocess



def varyParam(**kwargs):

    st = fst.getCurTimeObj()

    date_time = fst.getDateString()
    notes = kwargs.get('notes', '')
    N_runs = kwargs.get('N_runs', 1)
    show_plot = kwargs.get('show_plot', False)

    exclude_list = ['notes', 'N_runs', 'show_plot']
    vary_params, vary_param_dict_list, vary_param_tups = fst.parseSingleAndListParams(kwargs,exclude_list)

    label = 'vary_' + fst.listToFname(vary_params) + '_' + notes
    dir = fst.makeLabelDateDir(label)
    print('Saving vary param results to: ', dir)
    img_ext = '.png'
    base_fname = fst.combineDirAndFile(dir, label + date_time)
    img_fname = base_fname + img_ext

    log_fname = base_fname + '_log.txt'
    fst.writeDictToFile(kwargs, log_fname)

    # Set the SD for each entry to 0. If there's only 1 run each, that's fine. If
    # there are several, it will replace the 0's.
    R_tots = []
    SD = [0]*len(vary_param_dict_list)

    for i, kws in enumerate(vary_param_dict_list):

        print('\n{}\n'.format(vary_param_tups[i]))
        results = []
        for j in range(N_runs):
            print('run ',j)

            p1 = Population.Population(**kws, dir=dir, fname_notes=fst.paramDictToFnameStr(vary_param_tups[i]))

            _, _, r_tot = p1.evolve(**kws)

            results.append(r_tot)

        R_tots.append(mean(results))
        if N_runs>1:
            SD[i] = stdev(results)


    plt.close('all')
    fig,axes = plt.subplots(1,1,figsize=(6,9))

    plt.errorbar(list(range(len(R_tots))), R_tots, yerr=SD, fmt='ro-')

    axes.set_xticks(list(range(len(R_tots))))
    x_tick_labels = ['\n'.join(fst.dictToStringList(param)) for param in vary_param_tups]
    axes.set_xticklabels(x_tick_labels, rotation='vertical')
    axes.set_ylabel('Total reward')
    plt.tight_layout()
    plt.savefig(img_fname)

    vary_param_labels = [','.join(fst.dictToStringList(param)) for param in vary_param_tups]
    f = open(base_fname + '_values.txt','w+')
    for label, val, sd in zip(vary_param_labels, R_tots, SD):
        f.write('{}\t{}\t{}\n'.format(label, val, sd))
    f.close()

    print('\n\ntook {} to execute'.format(fst.getTimeDiffStr(st)))

    plotRewardCurvesByVaryParam(dir, searchlabel='bestscore')
    plotRewardCurvesByVaryParam(dir, searchlabel='meanscore')

    if show_plot:
        plt.show()




def plotRewardCurvesByVaryParam(dir, searchlabel, **kwargs):

    #Use the "values" file from now on to get the vary_param values
    # searchlabel will be the thing the fname we're searching for is prefaced with,
    # so we can do the same for multiple things.

    # Find the values file
    val_file_list = glob.glob(fst.addTrailingSlashIfNeeded(dir) + 'vary_' + '*' + 'values.txt')

    assert len(val_file_list)==1, 'there needs to be exactly one values.txt file.'

    vals_file = val_file_list[0]

    # Read in each line, corresponding to each vary params tuple
    with open(vals_file, 'r') as f:
        vary_param_vals = f.read().split('\n')

    # they're tab sep'd, so split and grab the first of each col.
    vary_param_vals = [x.split('\t')[0] for x in vary_param_vals if x!='']
    # Expects the vary params to be separated by underscores -- not ideal.
    vary_param_vals = [x.replace(',','_') for x in vary_param_vals]
    # Get the files that contain this series of vary vals...
    vary_param_files = [glob.glob(fst.addTrailingSlashIfNeeded(dir) + searchlabel + '*' + val + '*' + '.txt') for val in vary_param_vals]



    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    line_cols = ['darkred', 'mediumblue', 'darkgreen', 'goldenrod', 'purple', 'darkorange', 'black']
    shade_cols = ['tomato', 'dodgerblue', 'lightgreen', 'khaki', 'plum', 'peachpuff', 'lightgray']
    max_total = -1000
    min_total = 1000
    N_stds = 2
    N_skip = 0

    # This is a really hacky way of lining up curves that are shifted. You pass it a list of
    # how each curve (the avg) will be scaled and how each will be offset. If you don't pass it
    # anything, it won't do anything differently.
    scale_factors = kwargs.get('scale_factors', np.ones(len(vary_param_vals)))
    offsets = kwargs.get('offsets', np.zeros(len(vary_param_vals)))

    # For doing the scale and offset thing, and making sure the ranges are right.
    print('vary_param_vals', vary_param_vals)
    for i, (val, file_group) in enumerate(zip(vary_param_vals, vary_param_files)):
        dat_array = np.array([np.loadtxt(fname) for fname in file_group])
        avg = np.mean(dat_array, axis=0)*scale_factors[i] + offsets[i]
        std = np.std(dat_array, axis=0)*scale_factors[i]
        if max((avg + N_stds*std)[N_skip:]) > max_total:
            max_total = max((avg + N_stds*std)[N_skip:])
        if min((avg - N_stds*std)[N_skip:]) < min_total:
            min_total = min((avg - N_stds*std)[N_skip:])
        plt.plot(avg, color=line_cols[i], label=val)
        plt.fill_between(np.array(range(len(avg))), avg - std, avg + std, facecolor=shade_cols[i], alpha=0.5)

    #print(max_total, min_total)
    plt.legend()
    plt.xlabel('generations')
    plt.ylabel(searchlabel)
    plt.ylim((min_total,max_total))

    plt.savefig(fst.addTrailingSlashIfNeeded(dir) + 'all_' + searchlabel + '__'.join(vary_param_vals) + '__' + fst.getDateString() + '.png')

    # For each one separately
    print('vary_param_vals', vary_param_vals)
    for i, (val, file_group) in enumerate(zip(vary_param_vals, vary_param_files)):
        #print(max_total, min_total)
        plt.clf()
        dat_array = np.array([np.loadtxt(fname) for fname in file_group])
        avg = np.mean(dat_array, axis=0)*scale_factors[i] + offsets[i]
        std = np.std(dat_array, axis=0)*scale_factors[i]

        plt.plot(avg, color=line_cols[i], label=val)
        plt.legend()
        plt.xlabel('generations')
        plt.ylabel(searchlabel)
        plt.ylim((min_total,max_total))
        plt.fill_between(np.array(range(len(avg))), avg - std, avg + std, facecolor=shade_cols[i], alpha=0.5)
        plt.savefig(fst.addTrailingSlashIfNeeded(dir) + searchlabel + '__' + val + '__' + fst.getDateString() + '.png')



def plotPopulationProperty(dir, search_label, **kwargs):


    show_plot = kwargs.get('show_plot', False)
    save_plot = kwargs.get('save_plot', True)
    make_hist_gif = kwargs.get('make_hist_gif', True)

    # Find the values file
    prop_file_list = glob.glob(fst.addTrailingSlashIfNeeded(dir) + search_label + '*' + '.txt')

    assert len(prop_file_list)==1, 'there needs to be exactly one .txt file.'

    prop_file = prop_file_list[0]

    prop_dat = np.loadtxt(prop_file)

    N_gen = prop_dat.shape[0]
    max_gif_frames = 299
    gif_save_period = ceil(N_gen/max_gif_frames)

    avg = np.mean(prop_dat)
    std = np.std(prop_dat)

    print('dat min: {:.2f}'.format(np.min(prop_dat)))
    print('dat max: {:.2f}'.format(np.max(prop_dat)))
    print('dat mean: {:.2f}'.format(avg))
    print('dat std: {:.2f}'.format(std))

    dat_lb = max(np.min(prop_dat), avg - 2*std)
    dat_ub = min(np.max(prop_dat), avg + 2*std)

    dat_lims = (dat_lb - 0.1*abs(dat_lb), dat_ub + 0.1*abs(dat_ub))

    # Make histogram gif

    if make_hist_gif:

        gif_dir = fst.makeDir(fst.combineDirAndFile(dir, 'gif_imgs'))

        for i, gen_dat in enumerate(prop_dat):
            plt.clf()
            plt.hist(gen_dat, facecolor='dodgerblue', edgecolor='k', label=search_label, alpha=0.9, density=True)
            plt.axvline(np.mean(gen_dat), color='k', linestyle='dashed', linewidth=1)
            plt.xlim(dat_lims)
            plt.ylim((0, 1.0/len(gen_dat)))
            plt.title(f'generation {i}')
            plt.xlabel(search_label)
            plt.ylabel('counts')

            if save_plot:
                if i%gif_save_period == 0:
                    fname = fst.combineDirAndFile(gif_dir, f'{i}.png')
                    plt.savefig(fname)


        try:
            gif_name = fst.gifFromImages(gif_dir, f'{search_label}_hist', ext='.png', delay=5)
            print(gif_name)
            gif_basename = fst.fnameFromFullPath(gif_name)
            subprocess.check_call(['mv', gif_name, fst.combineDirAndFile(dir, gif_basename)])
            subprocess.check_call(['rm', '-rf', gif_dir])
        except:
            print('problem in creating gif')


    plt.clf()

    # Make time scatter plot

    # Right now this is in the structure where a row is a generation and each
    # entry of that row is an individ. We want to plot it per gen, so we need
    # to make it a set of (generation, value) points (so 5 gens of 8 individs would
    # go 5x8 -> 5x8x2 -> 40x2 -> 2x40)
    prop_pts = np.array([[[i, val] for val in prop_dat[i]] for i in range(len(prop_dat))])
    N_tot_entries = prop_dat.shape[0]*prop_dat.shape[1] # Makes it 5x8x2
    prop_pts = np.reshape(prop_pts, (N_tot_entries, 2)) # Makes it 40x2
    prop_pts = np.swapaxes(prop_pts, 0, 1) # Makes it 2x40

    plt.plot(prop_pts[0], prop_pts[1], 'o', color='dodgerblue')
    plt.xlabel('generations')
    plt.ylabel(search_label)
    plt.ylim(dat_lims)

    if save_plot:
        plt.savefig(fst.combineDirAndFile(dir, search_label + '_scatter_plot.png'))

    if show_plot:
        plt.show()

    plt.clf()

    # Make time std plot

    gen_mean = np.mean(prop_dat, axis=1)
    gen_std = np.std(prop_dat, axis=1)

    line_cols = ['darkred', 'mediumblue', 'darkgreen', 'goldenrod', 'purple', 'darkorange', 'black']
    shade_cols = ['tomato', 'dodgerblue', 'lightgreen', 'khaki', 'plum', 'peachpuff', 'lightgray']

    plt.fill_between(np.array(range(len(gen_mean))), gen_mean - gen_std, gen_mean + gen_std, facecolor=shade_cols[0], alpha=0.5)
    plt.plot(gen_mean, color=line_cols[0])

    plt.xlabel('generations')
    plt.ylabel(search_label)
    plt.ylim(dat_lims)

    if save_plot:
        plt.savefig(fst.combineDirAndFile(dir, search_label + '_mean-std_plot.png'))

    if show_plot:
        plt.show()

    plt.close()

















#
