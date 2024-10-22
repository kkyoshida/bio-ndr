import numpy as np
import model
import params

np.random.seed(2)

n_kc = 2000

def make_kc(data_index, prop=0.05, candidate_n=params.trial_test):
    data, label = model.data_load(data_index)

    kc_matrix_list = np.zeros((candidate_n, n_kc, np.shape(data)[0]))

    for j in range(candidate_n):
        # make random matrix
        w_matrix = np.zeros((n_kc, np.shape(data)[1]))
        index_list = np.array([i for i in range(np.shape(data)[1])])
        n_connection = 7
        for kc in range(n_kc):
            w_matrix[kc, np.random.choice(index_list, n_connection, replace=False)] = 1.0 

        # make KC
        kc_matrix = np.zeros((n_kc, np.shape(data)[0]))
        kc_matrix_temp = w_matrix @ data.T
        
        for i in range(np.shape(data)[0]):
            kenyon = kc_matrix_temp[:, i]
            kenyon[np.argsort(kenyon)[0 : int((1-prop) * n_kc)]] = 0 # Thresholding
            kc_matrix[:, i] = np.where(kenyon > 0, kenyon, 0)
     
        kc_matrix = kc_matrix / np.sum(kc_matrix, axis=0) # normalize in each input
        kc_matrix_list[j] = kc_matrix

    return kc_matrix_list   

kc_badel = make_kc(data_index=5)
np.save('../data/kc_badel', kc_badel)

# different sparseness for Hallem
data, label = model.data_load(data_index=4)
sparseness_list = params.sparseness_list
kc_hallem_sparseness = np.zeros((np.size(sparseness_list), params.trial_test, n_kc, np.shape(data)[0]))
for j in range(np.size(sparseness_list)):
    kc_hallem_sparseness[j] = make_kc(data_index=4, prop=sparseness_list[j])
np.savez_compressed('../data/kc_hallem_sparseness', kc_hallem_sparseness)

