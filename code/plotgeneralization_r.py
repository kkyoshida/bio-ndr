import numpy as np
import matplotlib.pyplot as plt
import model 
import figplot
from sklearn.linear_model import Perceptron

set_width = 0.5

np.random.seed(2)

# linear perceptron for original high-dimensional data
def lin_separate(rewardprop, r, data_index):
    data, ylabel = model.data_load(data_index=data_index)
    reward, rewardlabel, rewardstimlabel = model.set_reward(data_index=data_index, label=ylabel, reward_mode=rewardprop)
    ppn = Perceptron(random_state=r)
    
    data_select = data[np.where((rewardstimlabel==1) | (rewardstimlabel==-1))]
    rewardlabel_select = rewardlabel[np.where((rewardstimlabel==1) | (rewardstimlabel==-1))]
    ppn.fit(data_select, rewardlabel_select)
    
    w = ppn.coef_
    result = (w @ data.T)[0]
    result_aux = np.zeros((np.size(result), 2))
    result_aux[:, 0] = result

    return 100 * model.lin_separate_reward(result_aux, rewardlabel)

# linear perceptron after projecting to high dimensional space
def kc_separate(rewardprop, r, data_index, sigma):
    data, ylabel = model.data_load(data_index=data_index)
    reward, rewardlabel, rewardstimlabel = model.set_reward(data_index=data_index, label=ylabel, reward_mode=rewardprop)

    z = np.zeros((np.size(ylabel), np.size(ylabel)))
    for j in range(np.size(ylabel)):
        z[j] = np.exp(-np.sum((data - data[j].reshape(1, np.size(data[j])))**2, axis = 1)/(2 * sigma**2))
    
    ppn = Perceptron(random_state=r)
    z_select = z[np.where((rewardstimlabel==1) | (rewardstimlabel==-1))]
    rewardlabel_select = rewardlabel[np.where((rewardstimlabel==1) | (rewardstimlabel==-1))] 
    ppn.fit(z_select, rewardlabel_select)
    
    w = ppn.coef_
    result = (w @ z.T)[0]
    result_aux = np.zeros((np.size(result), 2))
    result_aux[:, 0] = result
    
    return 100 * model.lin_separate_reward(result_aux, rewardlabel)

fig = plt.figure(figsize=(7, 3.5))

for dataset in range(2):
    data_index = 1 + 2 * dataset  # 1: four ring or 3: s-curve

    if data_index == 1:
        perplexity = 20
    elif data_index == 3:
        perplexity = 40

    trial_number = 10
    sigma_number = 200
    result_kernel = np.zeros((10, trial_number, sigma_number))
    result_linear = np.zeros((10, trial_number))

    for i in range(10):
        for trial in range(trial_number):
            result_linear[i, trial] = lin_separate(0.1*(i+1), r=trial, data_index=data_index)
            for j in range(sigma_number):
                result_kernel[i, trial, j] = kc_separate(0.1*(i+1), r=trial, data_index=data_index, sigma=5.0*(j+1))
            
    #model
    data, ylabel = model.data_load(data_index=data_index)
    reward, rewardlabel, rewardstimlabel = model.set_reward(data_index=data_index, label=ylabel, reward_mode=1.0)
    result_model = np.zeros((10, trial_number))
    for i in range(10):
        for trial in range(trial_number):
            filename = '../data/rewardtsne{}-{}'.format(perplexity, 1000*data_index + 100*trial + i+1)
            w = np.load(filename + '.npz')['arr_0']
            result_model[i, trial] = 100 * model.lin_separate_reward(w, rewardlabel)
            
    x_index = np.array([0.1 * (i+1) for i in range(np.shape(result_linear)[0])])

    transform_sd = np.sqrt(trial_number)

    c_list = ['blue', 'red', 'green']
    ax = fig.add_subplot(2, 2, 1+dataset)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)

    kernel_index = np.argmax(np.sum(result_kernel, axis=(0, 1)))
    print(5.0 * (kernel_index + 1))

    result_kernel_plot = np.zeros((10, trial_number))
    for i in range(10):
        for trial in range(trial_number):
            result_kernel_plot[i, trial] = kc_separate(0.1*(i+1), r=trial+100, data_index=data_index, sigma=5.0*(kernel_index+1))

    ax.set_ylim(45, 105)
    ax.errorbar(x_index, np.mean(result_model, axis=1), yerr=np.std(result_model, axis=1, ddof=1)/transform_sd, capsize=3, color=c_list[0])
    ax.errorbar(x_index, np.mean(result_linear, axis=1), yerr=np.std(result_linear, axis=1, ddof=1)/transform_sd, capsize=3, color=c_list[1])
    ax.errorbar(x_index, np.mean(result_kernel_plot, axis=1), yerr=np.std(result_kernel_plot, axis=1, ddof=1)/transform_sd, capsize=3, color=c_list[2])
    ax.set_aspect(0.012)

plt.savefig("../figure/generalizationcomparison_v.pdf")
