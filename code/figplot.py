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

def plot_reward(filename, vmin, vmax):
    npz_comp = np.load('../data/' + filename + '.npz')
    w, label, rewardstimlabel = npz_comp['arr_0'], npz_comp['arr_5'], npz_comp['arr_6']
    newlabel = np.where(rewardstimlabel != 0, (vmax-vmin) / 2 * rewardstimlabel + (vmax+vmin) / 2, label)
    
    fig = plt.figure(figsize=(7, 3.5))

    ax = fig.add_subplot(2, 4, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.scatter(w[:, 0], w[:, 1], c=newlabel, cmap='jet', s=point_size, vmin=vmin, vmax=vmax)    

    plt.savefig('../figure/' + filename + '.pdf')


def plot_linsep_kl(filename, data): 
    trial_number = 5
    linear_separability_list = np.zeros((trial_number, 101))
    kl_divergence_list = np.zeros((trial_number, 101))
    for i in range(5):
        npz_comp = np.load('../data/' + filename + '{}.npz'.format(i))
        linear_separability_list[i], kl_divergence_list[i] = 100 * npz_comp['arr_1'], npz_comp['arr_3']

    fig = plt.figure(figsize=(7, 3.5))
    
    if data == 'MNIST':
        total_epoch = params.totalepoch_mnist
    elif data == 'ring':
        total_epoch = params.totalepoch_ring
    
    x_index = np.array([int(total_epoch/100) * i for i in range(101)])

    transform_sd = np.sqrt(trial_number)

    plot_x_index = int(500/int(total_epoch/100))

    ax = fig.add_subplot(2, 4, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.set_ylim(0, 1.1 * np.max(kl_divergence_list))
    ax.plot(x_index[plot_x_index: ], np.mean(kl_divergence_list, axis=0)[plot_x_index: ], linewidth=graph_width)
    standard_error_kl = np.std(kl_divergence_list, axis=0, ddof=1)/transform_sd
    ax.fill_between(x_index[plot_x_index: ], np.mean(kl_divergence_list, axis=0)[plot_x_index: ] + standard_error_kl[plot_x_index: ], np.mean(kl_divergence_list, axis=0)[plot_x_index: ] - standard_error_kl[plot_x_index: ], alpha=0.15)

    ax = fig.add_subplot(2, 4, 2)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.plot(x_index[plot_x_index: ], np.mean(linear_separability_list, axis=0)[plot_x_index: ], linewidth=graph_width)
    standard_error_linsep = np.std(linear_separability_list, axis=0, ddof=1)/transform_sd
    ax.fill_between(x_index[plot_x_index: ], np.mean(linear_separability_list, axis=0)[plot_x_index: ] + standard_error_linsep[plot_x_index: ], np.mean(linear_separability_list, axis=0)[plot_x_index: ] - standard_error_linsep[plot_x_index: ], alpha=0.15)
    
    if data == 'MNIST':
        ax.set_ylim(5, 85)
        linsep_mnist_pcatsne = np.load('../data/linsep_mnist_pcatsne.npy')
        ax.hlines(y=100*linsep_mnist_pcatsne[0], colors='red', xmin=x_index[plot_x_index], xmax=params.totalepoch_mnist, linestyles='dashed')
        ax.hlines(y=100*linsep_mnist_pcatsne[1], colors='blue', xmin=x_index[plot_x_index], xmax=params.totalepoch_mnist, linestyles='dashed')
    else:
        ax.set_ylim(35, 115)

    plt.savefig('../figure/' + filename + 'linsepkl.pdf')
    

def plot_timeseries(filename):
    npz_comp = np.load('../data/' + filename + '.npz')
    w, w_timecourse, label = npz_comp['arr_0'], npz_comp['arr_4'], npz_comp['arr_5']
    sigma_j, perplexity_count = npz_comp['arr_7'], npz_comp['arr_8']

    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(2, 4, 4)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.scatter(w[:, 0], w[:, 1], c=label, cmap='jet', s=point_size)    

    for i in range(3):
        ax = fig.add_subplot(2, 4, 1+i)
        ax.spines['right'].set_linewidth(set_width)
        ax.spines['left'].set_linewidth(set_width)
        ax.spines['top'].set_linewidth(set_width)
        ax.spines['bottom'].set_linewidth(set_width)
        ax.tick_params(width=set_width)
        ax.scatter(w_timecourse[i, :, 0], w_timecourse[i, :, 1], c=label, cmap='jet', s=point_size)

    plt.savefig('../figure/' + filename + '.pdf')

point_size_odor = 0.4

def plot_points(fig, X_f, y_label_f, p_x, p_y, p_n):
    ax = fig.add_subplot(p_x, p_y, p_n)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    plot_xmin = np.min(X_f[:, 0]) - 0.05 * (np.max(X_f[:, 0]) - np.min(X_f[:, 0]))
    plot_xmax = np.max(X_f[:, 0]) + 0.05 * (np.max(X_f[:, 0]) - np.min(X_f[:, 0]))
    plot_ymin = np.min(X_f[:, 1]) - 0.05 * (np.max(X_f[:, 1]) - np.min(X_f[:, 1]))
    plot_ymax = np.max(X_f[:, 1]) + 0.05 * (np.max(X_f[:, 1]) - np.min(X_f[:, 1]))
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)
    ax.scatter(X_f[:,0], X_f[:,1], c=y_label_f, cmap='jet', s=point_size_odor)
    ax.set_aspect(1 / (plot_ymax-plot_ymin) * (plot_xmax-plot_xmin))
    

    
