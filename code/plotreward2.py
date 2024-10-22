import figplot
import numpy as np
import matplotlib.pyplot as plt
import params

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['lines.linewidth'] = 0.75

set_width = 0.5
graph_width = 0.75
point_size = 0.1

def plot_rewsep_two(filename, data_index): 
    # Plot the time course of reward separbility
    trial_number = 10
    reward_separability_list = np.zeros((trial_number, 2, 101))
    rewp_list = np.array([1, 10])
   
    for i in range(trial_number):
        for rewp in range(2):
            npz_comp = np.load('../data/' + filename + '{}.npz'.format(1000*data_index + 100*i + rewp_list[rewp]))
            reward_separability_list[i, rewp] = 100 * npz_comp['arr_2']

    fig = plt.figure(figsize=(7, 3.5))

    if data_index == 1:
        total_epoch = 100000
    elif data_index == 3:
        total_epoch = 50000
    x_index = np.array([int(total_epoch/100) * i for i in range(101)])
    plot_x_index = int(500/int(total_epoch/100))
    transform_sd = np.sqrt(trial_number)
    
    c_list = ['purple', 'orange']
    ax = fig.add_subplot(2, 4, 1+rewp)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.set_ylim(45, 105)

    for rewp in range(2):
        ax.plot(x_index[plot_x_index: ], np.mean(reward_separability_list[:, rewp, :], axis=0)[plot_x_index: ], linewidth=graph_width, color=c_list[rewp])
        standard_error = np.std(reward_separability_list[:, rewp, :], axis=0, ddof=1) / transform_sd
        ax.fill_between(x_index[plot_x_index: ], np.mean(reward_separability_list[:, rewp, :], axis=0)[plot_x_index: ] + standard_error[plot_x_index: ], np.mean(reward_separability_list[:, rewp, :], axis=0)[plot_x_index: ] - standard_error[plot_x_index: ], alpha=0.15, color=c_list[rewp])
   
    plt.savefig('../figure/' + filename + 'rewseptwo{}.pdf'.format(data_index))

figplot.plot_reward('rewardtsne20-1005', vmin=-2, vmax=4.5)

plot_rewsep_two('rewardtsne20-', data_index=1)
plot_rewsep_two('rewardtsne40-', data_index=3)
