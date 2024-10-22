import numpy as np
import matplotlib.pyplot as plt
import figplot
import model

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['lines.linewidth'] = 0.75

point_size = 0.1
point_size_3d = 0.5
set_width = 0.5

np.random.seed(2)

# S-shaped curve
sshape, sshape_label = model.data_load(data_index=3)
fig = plt.figure(figsize=(12, 12))
reward, rewardlabel, rewardstimlabel = model.set_reward(3, sshape_label, reward_mode=0.1)
newlabel = np.where(rewardstimlabel !=0, 3.0 * rewardstimlabel + 1.0, sshape_label)
ax = fig.add_subplot(4, 4, 5, projection='3d')
ax.scatter(sshape[:, 0], sshape[:, 2], sshape[:, 1], c=newlabel, cmap='jet', s=point_size_3d*5, vmin=-2, vmax=4)
ax.view_init(elev=75, azim=-70)
plt.savefig('../figure/sshapeconnect3d.pdf')

# disconnected S-shaped curve
fig = plt.figure(figsize=(12, 12))
for i in range(2):
    sshape, sshape_label = model.data_load(data_index=6+i)
    reward, rewardlabel, rewardstimlabel = model.set_reward(6+i, sshape_label, reward_mode=0.1)
    newlabel = np.where(rewardstimlabel !=0, rewardstimlabel, 0.45)
    cmap = plt.get_cmap('jet')
    cmap_includegray = cmap(newlabel)
    if i==0: 
        cmap_includegray[int(0.0625 * np.size(sshape_label)): int(0.50 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.7)
        cmap_includegray[int(0.75 * np.size(sshape_label)): int(0.9375 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.7)
    elif i==1:
        cmap_includegray[int(0.0625 * np.size(sshape_label)): int(0.25 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.7)
        cmap_includegray[int(0.50 * np.size(sshape_label)): int(0.9375 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.7)
    ax = fig.add_subplot(4, 4, 1+2*i, projection='3d')
    ax.scatter(sshape[:, 0], sshape[:, 2], sshape[:, 1], color=cmap_includegray, s=point_size_3d*5)
    ax.view_init(elev=75, azim=-70)
plt.savefig('../figure/sshapedisc3d.pdf')

def plot_reward(filename, data_index):
    npz_comp = np.load('../data/' + filename + '.npz')
    w, label, rewardstimlabel = npz_comp['arr_0'], npz_comp['arr_5'], npz_comp['arr_6']
    newlabel = np.where(rewardstimlabel != 0, rewardstimlabel, 0.45)
    cmap = plt.get_cmap('jet')
    cmap_includegray = cmap(newlabel)
    if data_index==6: 
        cmap_includegray[int(0.0625 * np.size(sshape_label)): int(0.50 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.3)
        cmap_includegray[int(0.75 * np.size(sshape_label)): int(0.9375 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.3)
    elif data_index==7:
        cmap_includegray[int(0.0625 * np.size(sshape_label)): int(0.25 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.3)
        cmap_includegray[int(0.50 * np.size(sshape_label)): int(0.9375 * np.size(sshape_label))] = (0.5, 0.5, 0.5, 0.3)
    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(2, 4, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.tick_params(width=set_width)
    ax.scatter(w[:, 0], w[:, 1], color=cmap_includegray, s=point_size)    
    plt.savefig('../figure/' + filename + '.pdf')

plot_reward('rewardtsne40-sshapedisc-6', data_index=6)
plot_reward('rewardtsne40-sshapedisc-7', data_index=7)