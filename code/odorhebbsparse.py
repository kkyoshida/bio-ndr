import numpy as np
import model
import params

np.random.seed(2)

data_kc_sparseness = np.load('../data/kc_hallem_sparseness.npz')['arr_0']

for i in range(np.shape(data_kc_sparseness)[0]):
    for trial in range(np.shape(data_kc_sparseness)[1]):
        data_kc = data_kc_sparseness[i, trial].T
        model.hebbtsne(data_index=4, epoch_number=params.totalepoch_odor, filename='hallemtsne-sparse20-{}'.format(100*i+trial), x_to_z=data_kc, winner_take_all=False, set_perplexity=20.0) 
        model.hebbtsne(data_index=4, epoch_number=params.totalepoch_odor, filename='hallemtsne-sparse30-{}'.format(100*i+trial), x_to_z=data_kc, winner_take_all=False, set_perplexity=30.0) 
