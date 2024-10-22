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

def plot_sigma_perp(filename, data): 
    trial_number = 5

    sigma_j_list = np.zeros((trial_number, 1001))
    perplexity_count_list = np.zeros((trial_number, 1001))

    for i in range(5):
        npz_comp = np.load('../data/' + filename + '{}.npz'.format(i))
        sigma_j_list[i], perplexity_count_list[i] =  npz_comp['arr_7'], npz_comp['arr_8']

    fig = plt.figure(figsize=(7, 3.5))
    
    x_index = np.array([i for i in range(1001)])

    transform_sd = np.sqrt(trial_number)

    ax = fig.add_subplot(2, 4, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    standard_error_sigma_j = np.std(sigma_j_list, axis=0, ddof=1)/transform_sd
    ax.set_ylim(0, 1.1 * np.max(np.mean(sigma_j_list, axis=0) + standard_error_sigma_j))
    ax.plot(x_index, np.mean(sigma_j_list, axis=0), linewidth=graph_width)
    ax.fill_between(x_index, np.mean(sigma_j_list, axis=0) + standard_error_sigma_j, np.mean(sigma_j_list, axis=0) - standard_error_sigma_j, alpha=0.15)

    ax = fig.add_subplot(2, 4, 2)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    standard_error_perp = np.std(perplexity_count_list, axis=0, ddof=1)/transform_sd
    ax.set_ylim(0, 1.1 * np.max(np.mean(perplexity_count_list, axis=0) + standard_error_perp))
    ax.plot(x_index, np.mean(perplexity_count_list, axis=0), linewidth=graph_width)
    ax.fill_between(x_index, np.mean(perplexity_count_list, axis=0) + standard_error_perp, np.mean(perplexity_count_list, axis=0) - standard_error_perp, alpha=0.15)

    plt.savefig('../figure/' + filename + 'sigmaperp.pdf')

plot_sigma_perp('ringtsne20-', data='ring')
plot_sigma_perp('mnist40-', data='MNIST')
    

    
