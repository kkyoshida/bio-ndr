import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import squareform
from sklearn import svm

def lin_separate(data, label): 
    #linear separability by linear SVM
    ppn = svm.SVC(kernel='linear', random_state=0) 
    ppn.fit(data, label)
    pred = ppn.predict(data)
    return accuracy_score(label, pred)

def lin_separate_reward(data, rewardlabel):
    #reward separability
    data_reward = data[rewardlabel==1]
    data_notreward = data[rewardlabel==0]
    order_reward = np.argsort(data_reward[:, 0]) #sort rewarded data by x_0 axis
    order_notreward = np.argsort(data_notreward[:, 0]) #sort not-rewarded data by x_0 axis
    index_r = 0
    index_n = 0
    result = np.zeros(np.shape(data)[0]+1)
    result[0] = np.shape(data_reward)[0] / np.shape(data)[0]
    #increase threshold value, and calculate the score for each threshold
    for i in range(np.shape(data)[0]):
        if index_r < np.shape(data_reward)[0] and index_n < np.shape(data_notreward)[0]:
            if data_reward[order_reward[index_r], 0] <= data_notreward[order_notreward[index_n], 0]:
                index_r += 1
            else:
                index_n += 1
        elif index_r == np.shape(data_reward)[0]: 
            index_n += 1
        elif index_n == np.shape(data_notreward)[0]: 
            index_r += 1
        else:
            print('error')
        result[index_r + index_n] = (index_n - index_r + np.shape(data_reward)[0]) / np.shape(data)[0]
    return np.max(result)

def data_load(data_index):
    # load each data
    if data_index == 0:
        data = np.load('../data/ring1.npy')
        label = np.load('../data/ring1_label.npy')
    elif data_index == 1:
        data = np.load('../data/ring2.npy')
        label = np.load('../data/ring2_label.npy')
    elif data_index == 2: 
        data = np.load('../data/mnist.npy')
        label = np.load('../data/mnist_label.npy')
    elif data_index == 3:
        data = np.load('../data/sshape.npy')
        label = np.load('../data/sshape_label.npy')
    elif data_index == 4:
        a = np.loadtxt('../data/odorspaceHallem2006.csv', delimiter=",", dtype = "unicode",  encoding='utf-8-sig')
        b = a.astype(float)
        data = b[1:111, 1:25]
        label = b[1:111, 0]
    elif data_index == 5:
        a = np.loadtxt('../data/odordata_Badel2016.csv', delimiter=",", dtype = "unicode",  encoding='utf-8-sig')
        b = a.astype(float)
        data = b[1:np.shape(b)[0], 2:np.shape(b)[1]]
        label = b[1:np.shape(b)[0], 1]
    elif data_index == 6:
        data_original = np.load('../data/sshape.npy')
        label_original = np.load('../data/sshape_label.npy')
        data = np.delete(data_original, np.arange(int(0.6 * np.size(label_original)), int(0.8 * np.size(label_original))), axis=0)
        label = np.delete(label_original, np.arange(int(0.6 * np.size(label_original)), int(0.8 * np.size(label_original))))
    elif data_index == 7:
        data_original = np.load('../data/sshape.npy')
        label_original = np.load('../data/sshape_label.npy')
        data = np.delete(data_original, np.arange(int(0.2 * np.size(label_original)), int(0.4 * np.size(label_original))), axis=0)
        label = np.delete(label_original, np.arange(int(0.2 * np.size(label_original)), int(0.4 * np.size(label_original))))
    return data, label

def compute_distance_low(data_matrix):
    # Calculate the output similarity in t-SNE
    distance_matrix = np.zeros((np.size(data_matrix[:, 0]), np.size(data_matrix[:, 0])))
    for i in range(np.size(data_matrix[:, 0])):
        distance_matrix[i] = (1 + np.sum((data_matrix[i, :] - data_matrix) ** 2, axis=1)) ** (-1)
        distance_matrix[i, i] = 0.0
    return distance_matrix / np.sum(distance_matrix)

def evaluate_kl(w, p_matrix): 
    # Calculate the KL divergence between input and output similarity in t-SNE
    q = compute_distance_low(w) 
    kl_div = np.sum((p_matrix[p_matrix != 0] * np.log(p_matrix[p_matrix != 0] / q[p_matrix != 0])))
    return kl_div

def set_reward(data_index, label, reward_mode):
    # set the reward
    reward = np.zeros(np.size(label))
    rewardlabel = np.zeros(np.size(label))
    rewardstimlabel = np.zeros(np.size(label))
    if reward_mode == 0:
        reward_strength = 0.0
    else:
        reward_strength = 0.0000075 / reward_mode
    if data_index == 1:
        point_per_ring = int(0.25 * np.size(label))
        datanum = int(point_per_ring * reward_mode)
        for ring_i in range(4):
            if ring_i%2 == 0:
                reward_sign = 1
            else:
                reward_sign = -1
            reward[ring_i * point_per_ring  : datanum + ring_i * point_per_ring] = reward_sign * reward_strength
            rewardstimlabel[ring_i * point_per_ring  : datanum + ring_i * point_per_ring] = reward_sign     
        rewardlabel = 1 - label%2
    elif data_index == 3:
        reward[0 : int(0.5 * np.size(label) * reward_mode) ] = reward_strength
        reward[np.size(label) - int(0.5 * np.size(label) * reward_mode) : np.size(label) ] = -reward_strength
        rewardstimlabel[0 : int(0.5 * np.size(label) * reward_mode) ] = 1
        rewardstimlabel[np.size(label) - int(0.5 * np.size(label) * reward_mode) : np.size(label)] = -1
        rewardlabel = 1 - label%2
    elif data_index == 6 or data_index == 7:
        reward[0 : int(0.625 * np.size(label) * reward_mode) ] = reward_strength
        reward[np.size(label) - int(0.625 * np.size(label) * reward_mode) : np.size(label) ] = -reward_strength
        rewardstimlabel[0 : int(0.625 * np.size(label) * reward_mode) ] = 1
        rewardstimlabel[np.size(label) - int(0.625 * np.size(label) * reward_mode) : np.size(label)] = -1
        rewardlabel = 1 - label%2
    return reward, rewardlabel, rewardstimlabel

def div_include_zero(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b!=0))

def hebbtsne(data_index, epoch_number, filename, set_perplexity, x_to_z=0, reward_mode=0, winner_take_all=True, simple_option=False, calculate_linsep=False, calculate_kl=False):
    # Conduct Hebbian t-SNE
    data, label = data_load(data_index)
    sigma_j_initial = 500.0 # initial value of sigma_j
    learning_rate = 0.1
    unit_epoch = int(np.size(label) * (np.size(label)-1) / 10)

    reward, rewardlabel, rewardstimlabel = set_reward(data_index, label, reward_mode)

    datasize = np.size(data[:, 0])
    component = 2 # dimension of output y

    if winner_take_all:
        z_fix = np.identity(datasize)
    else:
        z_fix = x_to_z

    a_fix = np.zeros((np.shape(z_fix))) # axonal activity
    for j in range(np.shape(z_fix)[1]):
        max_index = np.where(z_fix[:, j] == np.max(z_fix[:, j]))[0]
        a_fix[np.random.choice(max_index), j] = 1
    a_fix = np.where(z_fix != 0, 1, 0) * a_fix 

    # Adam
    beta_1 = 0.9
    beta_2 = 0.999
    gradi_average = np.zeros((np.shape(z_fix)[1], component))
    s_average = np.zeros((np.shape(z_fix)[1], component))
    epsi = 0.00000001*np.ones((np.shape(z_fix)[1], component))

    # time constants
    tau_ydiff = 100.0
    tau_xdiff = 100.0

    # Regulating sigma
    perplexity_target = set_perplexity
    perplexity_count = np.log(perplexity_target) / np.log(2) * np.ones(datasize) 
    epsi_forlog = 0.00000001
    sigma_j_log = np.log(sigma_j_initial) * np.ones(np.shape(z_fix)[1])
    sigma_j = np.exp(sigma_j_log)
    tau_perp = 100.0

    if simple_option:
        sigma_j_all_log = np.log(sigma_j_initial)
        sigma_j_all = sigma_j_initial
        perplexity_count_simple = np.log(perplexity_target) / np.log(2)

    if calculate_kl:
        distances = np.array([np.sum((data[i, :] - data) ** 2, axis=1) for i in range(data.shape[0])])
        p_matrix = squareform(_joint_probabilities(distances, desired_perplexity=perplexity_target, verbose=0))

    # initial synaptic weights
    w = np.random.randn(np.shape(z_fix)[1], component) # standard normal distribution

    previous_datashow = np.random.randint(0, datasize)
    previous_y = z_fix[previous_datashow] @ w
    epsi_xinitial = 0.00000001

    plot_epoch = int(epoch_number/100)
    plot_epoch_initial = 1000

    linear_separability = np.zeros(1 + int(epoch_number/plot_epoch))
    reward_separability = np.zeros(1 + int(epoch_number/plot_epoch))
    kl_divergence = np.zeros(1 + int(epoch_number/plot_epoch))
    
    sigma_j_timecourse = np.zeros(1 + plot_epoch_initial)
    perplexity_count_timecourse = np.zeros(1 + plot_epoch_initial) 

    if calculate_linsep:
        # initial linear_separability and KL divergence
        linear_separability[0] = lin_separate(z_fix @ w, label)
        reward_separability[0] = lin_separate_reward(z_fix @ w, rewardlabel)
        if calculate_kl:
            kl_divergence[0] = evaluate_kl(z_fix @ w, p_matrix)
        if simple_option == False:
            sigma_j_timecourse[0] = sigma_j[0]
    
    w_timecourse = np.zeros((3, datasize, component))
    w_timecourse[0] = z_fix @ w   

    for epoch in range(epoch_number):
        datashow = np.zeros(unit_epoch+1, dtype=int)
        datatransit = np.random.randint(1, datasize, unit_epoch)
        datashow[0] = previous_datashow

        z = np.zeros((unit_epoch+1, np.shape(z_fix)[1]))
        a_epoch = np.zeros((unit_epoch, np.shape(z_fix)[1])) 
        z[0] = z_fix[datashow[0]]
        reward_timeseries = np.zeros(unit_epoch)

        for t in range(unit_epoch):
            datashow[t+1] = np.mod(datashow[t] + datatransit[t], datasize) # data shown in random order meeting datashow[t+1]!=datashow[t]
            z[t+1] = z_fix[datashow[t+1]]
            a_epoch[t] = a_fix[datashow[t+1]]
            reward_timeseries[t] = reward[datashow[t+1]] 
        
        distance = np.exp((np.sum(-(data[datashow[1:(unit_epoch+1)], :] - data[datashow[0:unit_epoch], :]) ** 2, axis=1).reshape([unit_epoch, 1])) @ (1/(2 * sigma_j**2).reshape([1, np.shape(z_fix)[1]])))
        y_now = z @ w
        y_now[0] = previous_y
        y_diff = 1/ (1 + np.sum((y_now[1:(unit_epoch+1)] - y_now[0:(unit_epoch)])**2, axis=1))
        x_diff_acti_now = a_epoch * distance 
        if simple_option:
            x_diff_sum_simple = np.exp(np.sum(- (data[datashow[1:(unit_epoch+1)], :] - data[datashow[0:unit_epoch], :]) ** 2, axis=1)/(2 * sigma_j_all**2))
        
        if epoch == 0:
            # set the initial value of x_diff_average and y_diff_average
            x_diff_average = epsi_xinitial + (datasize-1) * div_include_zero(np.sum(a_epoch[0:unit_epoch] * np.sum(x_diff_acti_now, axis=1)[:, np.newaxis], axis=0), np.sum(a_epoch[0:unit_epoch], axis=0))
            if simple_option:
                x_diff_sum_simple_average = epsi_xinitial + np.mean(x_diff_sum_simple) * (datasize - 1)
            y_diff_average = np.mean(y_diff, axis=0) * datasize * (datasize - 1)
       
        x_diff_sum = np.sum(div_include_zero(x_diff_acti_now, x_diff_average), axis=1)
        if simple_option: 
            x_diff_sum = x_diff_sum_simple/x_diff_sum_simple_average

        if epoch == 0:
            # set the initial value of perplexity_count
            perplexity_count = div_include_zero(-np.sum((datasize-1) * a_epoch[0:unit_epoch] * (x_diff_sum * np.log(x_diff_sum + epsi_forlog) / np.log(2))[:, np.newaxis], axis=0), np.sum(a_epoch[0:unit_epoch], axis=0))
            if simple_option:
                perplexity_count_simple = - np.mean((datasize-1) * x_diff_sum * np.log(x_diff_sum + epsi_forlog) / np.log(2))
            if calculate_linsep:
                if simple_option == False:
                    perplexity_count_timecourse[0] = 2 ** perplexity_count[0]

        if epoch >= 500:
            # Calculate synaptic changes
            aux1 = (-2 * (-y_diff/y_diff_average + x_diff_sum/datasize) * y_diff).reshape(unit_epoch, 1) @ np.ones((1, component))
            aux2 = (z[1:(unit_epoch+1)] - z[0:unit_epoch]).reshape((unit_epoch, np.shape(z_fix)[1], 1)) @ (y_now[1:(unit_epoch+1)] - y_now[0:(unit_epoch)]).reshape((unit_epoch, 1, component))
            gradisum = datasize * (datasize-1) / unit_epoch * np.sum(np.expand_dims(aux1, 1) * aux2, axis=0) #the sum of gradient in one unitepoch
            gradisum[:, 0] += 1.0 * datasize / unit_epoch * np.sum(z[1:(unit_epoch+1)] * reward_timeseries[0:unit_epoch].reshape(unit_epoch, 1), axis=0) #reward_modulation
            gradi_average = beta_1 * gradi_average + (1.0 - beta_1) * gradisum
            s_average = beta_2 * s_average + (1.0 - beta_2) * gradisum **2
            w += learning_rate * gradi_average / (1.0 - beta_1**(epoch-499)) / ((s_average/(1.0-beta_2**(epoch-499)))**(1/2) + epsi)
        
        y_diff_average = y_diff_average + 1/tau_ydiff * (-y_diff_average + np.mean(y_diff, axis=0) * datasize * (datasize - 1))
        x_diff_average = x_diff_average + 1/tau_xdiff * x_diff_average * div_include_zero(np.sum(a_epoch[0:unit_epoch] * (-1 + (datasize - 1) * x_diff_sum[:, np.newaxis]), axis=0), np.sum(a_epoch[0:unit_epoch], axis=0))
        perplexity_count = perplexity_count + 1/tau_perp * (-perplexity_count - div_include_zero(np.sum((datasize - 1) * a_epoch[0:unit_epoch] * (x_diff_sum * np.log(x_diff_sum + epsi_forlog) / np.log(2))[:, np.newaxis], axis=0), np.sum(a_epoch[0:unit_epoch], axis=0)))
        if simple_option:
            x_diff_sum_simple_average = x_diff_sum_simple_average + 1/tau_xdiff * (-x_diff_sum_simple_average + np.mean(x_diff_sum_simple) * (datasize - 1))
            perplexity_count_simple = perplexity_count_simple + 1/tau_perp * (-perplexity_count_simple - np.mean((datasize-1)* x_diff_sum * np.log(x_diff_sum + epsi_forlog) / np.log(2)))

        if simple_option == False:
            sigma_j_log += -0.001 * (2**perplexity_count - perplexity_target) * np.where(np.sum(a_fix, axis=0) != 0, 1, 0) # If np.sum(a_fix, axis=0)==0, the axon is active for no inputs. 
            sigma_j = np.exp(sigma_j_log)
        else:
            sigma_j_all_log += -0.001 * (2**perplexity_count_simple - perplexity_target)
            sigma_j_all = np.exp(sigma_j_all_log)
       
        previous_datashow = np.copy(datashow[-1])
        previous_y = np.copy(y_now[-1])

        if calculate_linsep:
            if epoch%plot_epoch == plot_epoch - 1:
                linear_separability[1 + int(epoch/plot_epoch)] = lin_separate(z_fix @ w, label)
                reward_separability[1 + int(epoch/plot_epoch)] = lin_separate_reward(z_fix @ w, rewardlabel)
                if calculate_kl:
                    kl_divergence[1 + int(epoch/plot_epoch)] = evaluate_kl(z_fix @ w, p_matrix)

            if epoch < plot_epoch_initial:
                if simple_option==False:
                    sigma_j_timecourse[1 + epoch] = sigma_j[0]
                    perplexity_count_timecourse[1 + epoch] = 2 ** perplexity_count[0]
                
        # timecourse of the representation
        if epoch == 699:
            w_timecourse[1] = z_fix @ w
        elif epoch == 999:
            w_timecourse[2] = z_fix @ w
        
    np.savez_compressed('../data/' + filename, z_fix @ w, linear_separability, reward_separability, kl_divergence, w_timecourse, label, rewardstimlabel, sigma_j_timecourse, perplexity_count_timecourse, w)
   
