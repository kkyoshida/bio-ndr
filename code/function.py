import numpy as np
import time
import model

# MNIST-simulation
def hebbtsne_mnist(epoch_number, seed, simple_option):
    t1 = time.time() 
    np.random.seed(seed)
    if simple_option == False:
        model.hebbtsne(data_index=2, epoch_number=epoch_number, filename='mnist40-{}'.format(seed), calculate_linsep=True, calculate_kl=True, set_perplexity=40.0)
    else:
        model.hebbtsne(data_index=2, epoch_number=epoch_number, filename='mnist_simple40-{}'.format(seed), calculate_linsep=True, calculate_kl=True, simple_option=True, set_perplexity=40.0)
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Total time：{elapsed_time}")

# reward-simulation
def hebbtsne_reward(data_index, epoch_number, seed, mode):
    t1 = time.time()
    if mode == 0:
        np.random.seed(seed)
    elif mode == 1:
        np.random.seed(20 + seed)
    for i in range([0, 5][mode], [5, 11][mode]):
        if i==0 or i==1 or i==10:
            if data_index == 1:
                model.hebbtsne(data_index=data_index, epoch_number=epoch_number, filename='rewardtsne20-{}'.format(1000*data_index + 100*seed + i), reward_mode=0.1*i, calculate_linsep=True, set_perplexity=20.0)
            elif data_index == 3:
                model.hebbtsne(data_index=data_index, epoch_number=epoch_number, filename='rewardtsne40-{}'.format(1000*data_index + 100*seed + i), reward_mode=0.1*i, calculate_linsep=True, set_perplexity=40.0)
        else:
            if data_index == 1:
                model.hebbtsne(data_index=data_index, epoch_number=epoch_number, filename='rewardtsne20-{}'.format(1000*data_index + 100*seed + i), reward_mode=0.1*i, set_perplexity=20.0)
            elif data_index == 3:
                model.hebbtsne(data_index=data_index, epoch_number=epoch_number, filename='rewardtsne40-{}'.format(1000*data_index + 100*seed + i), reward_mode=0.1*i, set_perplexity=40.0)
            
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Total time：{elapsed_time}")