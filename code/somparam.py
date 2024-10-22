# This file contains modified code from https://github.com/FlorentF9/SOMperf in the following paper:  
# Forest, Florent, Mustapha Lebbah, Hanane Azzag, and Jérôme Lacaille (2020). 
# A Survey and Implementation of Performance Metrics for Self-Organized Maps. arXiv, November 11, 2020. 
# https://doi.org/10.48550/arXiv.2011.05847.

import numpy as np
import model
from minisom import MiniSom
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

np.random.seed(2)

def Kaski_Lagus_error(som, input_data, som_x):
    weights = som.get_weights()
    weights = weights.reshape(-1, weights.shape[2])

    d = np.array([[np.sqrt(np.sum((input_data[k] - weights[l]) ** 2)) for l in range(weights.shape[0])] for k in range(input_data.shape[0])])
    
    neighbor = np.zeros((som_x ** 2, som_x ** 2)) 
    for i in range(som_x ** 2):
        if i + som_x < som_x ** 2:
            neighbor[i, i+som_x] = 1
            neighbor[i+som_x, i] = 1
        if i % som_x != som_x-1:
            neighbor[i, i+1] = 1
            neighbor[i+1, i] = 1
        
    d_som = csr_matrix([[np.sqrt(np.sum(np.square(weights[k] - weights[l]))) if neighbor[k, l] == 1 else np.inf
                        for l in range(weights.shape[0])]
                        for k in range(weights.shape[0])])

    tbmus = np.argsort(d, axis=1)[:, :2]  # two best matching units
    ces = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        ces[i] = d[i, tbmus[i, 0]]
        if neighbor[tbmus[i, 0], tbmus[i, 1]] == 1:  # if BMUs are neighbors
            ces[i] += d_som[tbmus[i, 0], tbmus[i, 1]]
        else:
            ces[i] += shortest_path(csgraph=d_som, method='auto', directed=False, return_predecessors=False, indices=tbmus[i, 0])[tbmus[i, 1]]
    return np.mean(ces)

for data_index in range(3):
    data, data_label = model.data_load(data_index=[2, 4, 5][data_index])
    
    map_size = np.rint(np.sqrt(5*np.sqrt(np.size(data_label)))).astype(int)
    print(map_size)

    best_error = np.inf  
    
    trial_number = 10
    
    for sigma_index in range(5):
        sigma = [1.0, 2.0, 4.0, 8.0, 16.0][sigma_index]
        print(sigma)
        for lr_index in range(7):
            lr = 2.0 * (0.5 ** (lr_index+1))
   
            kl_error = np.zeros(trial_number)
            for r in range(trial_number):
                som = MiniSom(x=map_size, y=map_size, input_len=data.shape[1],
                            sigma=sigma, learning_rate=lr, random_seed=r+100)
                som.random_weights_init(data)
                som.train_random(data, 10000)  

                kl_error[r] = Kaski_Lagus_error(som, data, map_size)

            if np.mean(kl_error) < best_error:
                best_error = np.mean(kl_error)
                best_params = np.array([sigma, lr])

    print(best_params)
    np.save('../data/best_sigma_lr{}'.format([2, 4, 5][data_index]), best_params)
   