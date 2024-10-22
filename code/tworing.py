import numpy as np
import time
import model
import params

#Fig.2: ring
t1 = time.time() 

np.random.seed(1)
trial_number = 5
total_epoch = params.totalepoch_ring

for j in range(trial_number):
    model.hebbtsne(data_index=0, epoch_number=total_epoch, filename='ringtsne20-{}'.format(j), calculate_linsep=True, calculate_kl=True, set_perplexity=20.0)
    model.hebbtsne(data_index=0, epoch_number=total_epoch, filename='ringtsne_simple20-{}'.format(j), calculate_linsep=True, calculate_kl=True, simple_option=True, set_perplexity=20.0)
    
t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
