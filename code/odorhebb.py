import numpy as np
import model
import params

np.random.seed(2)

for j in range(params.trial_test):
    data_kc = np.load('../data/kc_badel.npy')[j].T
    model.hebbtsne(data_index=5, epoch_number=params.totalepoch_odor, filename='badeltsne20-{}'.format(j), x_to_z=data_kc, winner_take_all=False, set_perplexity=20.0)
    model.hebbtsne(data_index=5, epoch_number=params.totalepoch_odor, filename='badeltsne30-{}'.format(j), x_to_z=data_kc, winner_take_all=False, set_perplexity=30.0)  