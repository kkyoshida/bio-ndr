import numpy as np
import time
import model

# reward-simulation-sshapedisc
t1 = time.time()

np.random.seed(2)

model.hebbtsne(data_index=7, epoch_number=20000, filename='rewardtsne40-sshapedisc-7', reward_mode=0.1, set_perplexity=40.0)
        
t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")