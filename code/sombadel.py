import numpy as np
import model
from minisom import MiniSom

np.random.seed(2)

data, ylabel = model.data_load(data_index=5)

m_size = 7
best_sigma = np.load('../data/best_sigma_lr5.npy')[0]
best_lr = np.load('../data/best_sigma_lr5.npy')[1]

trial_number = 10
som_result_tot = np.zeros((trial_number, np.size(ylabel), 2))

for trial in range(trial_number):
    som = MiniSom(x=m_size, y=m_size, input_len=np.shape(data)[1], sigma=best_sigma, learning_rate=best_lr, random_seed=trial)
    som.random_weights_init(data)
    som.train_random(data, 10000) 

    som_result = np.zeros((np.size(ylabel), 2))

    for i, x in enumerate(data):
        w = som.winner(x)  
        som_result[i] = w

    som_result_tot[trial] = som_result

np.save('../data/som_result_badel', som_result_tot)

