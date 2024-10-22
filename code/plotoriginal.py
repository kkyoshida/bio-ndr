import numpy as np
import matplotlib.pyplot as plt
import figplot
import model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(2)

ring1, ring1_label = model.data_load(data_index=0)
ring2, ring2_label = model.data_load(data_index=1)
mnist, mnist_label = model.data_load(data_index=2)
sshape, sshape_label = model.data_load(data_index=3)

point_size = 0.1
point_size_3d = 0.5
set_width = 0.5

# original data
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(4, 4, 1, projection='3d')
ax.scatter(ring1[:, 0], ring1[:, 1], ring1[:, 2], c=ring1_label, cmap='jet', s=point_size_3d)

for p in range(2):
    reward, rewardlabel, rewardstimlabel = model.set_reward(1, ring2_label, reward_mode=[0.10, 0][p])
    newlabel = np.where(rewardstimlabel !=0, 3.25 * rewardstimlabel + 1.25, ring2_label)
    ax = fig.add_subplot(4, 4, p+2, projection='3d')
    ax.view_init(elev=20, azim=-70)
    ax.scatter(ring2[:, 0], ring2[:, 1], ring2[:, 2], c=newlabel, cmap='jet', s=point_size_3d, vmin=-2.0, vmax=4.5)

reward, rewardlabel, rewardstimlabel = model.set_reward(3, sshape_label, reward_mode=0.1)
newlabel = np.where(rewardstimlabel !=0, 3.0 * rewardstimlabel + 1.0, sshape_label)

plt.savefig('../figure/ringplot.pdf')


# S-shaped curve, pca, and t-sne
fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(4, 4, 1)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
pca_ring = PCA(n_components=2, svd_solver='full', random_state=1)
pca_ring.fit(ring1)
Xd_ring = pca_ring.transform(ring1)
plot_ring = ax.scatter(Xd_ring[:,0], Xd_ring[:,1], c=ring1_label, cmap='jet', s=point_size)

ax = fig.add_subplot(4, 4, 5)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
pca_mnist = PCA(n_components=2, svd_solver='full', random_state=1)
pca_mnist.fit(mnist)
Xd_mnist = pca_mnist.transform(mnist)
plot_mnist = ax.scatter(Xd_mnist[:,0], Xd_mnist[:,1], c=mnist_label, cmap='jet', s=point_size)

# t-sne using sklearn
tsne = TSNE(n_components=2, random_state=1, perplexity=20)
ring_reduced = tsne.fit_transform(ring1)
ax = fig.add_subplot(4, 4, 2)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
plot_ring_tsne = ax.scatter(ring_reduced[:, 0], ring_reduced[:, 1], c=ring1_label, cmap='jet', s=point_size)

tsne = TSNE(n_components=2, random_state=1, perplexity=40)
mnist_reduced = tsne.fit_transform(mnist)
ax = fig.add_subplot(4, 4, 6)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
plot_mnist_tsne = ax.scatter(mnist_reduced[:, 0], mnist_reduced[:, 1], c=mnist_label, cmap='jet', s=point_size)

plt.savefig('../figure/pca_tsne.pdf')

# MNIST sample
fig = plt.figure(figsize=(12, 4.8))
index = [16, 8, 1, 6, 18, 3, 0, 2, 22, 17] # corresponding from zero to nine
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(mnist[index[i]].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.savefig('../figure/mnistsample.pdf')

# linear separability
trial_number = 5
linsep_mnist_pca = model.lin_separate(Xd_mnist, mnist_label)
linsep_mnist_tsne_40 = np.zeros(trial_number)

for j in range(5):
    tsne = TSNE(n_components=2, random_state=j, perplexity=40)
    mnist_reduced = tsne.fit_transform(mnist)
    linsep_mnist_tsne_40[j] = model.lin_separate(mnist_reduced, mnist_label)

print(linsep_mnist_pca, np.mean(linsep_mnist_tsne_40))
linsep = np.array([linsep_mnist_pca, np.mean(linsep_mnist_tsne_40)])
np.save('../data/linsep_mnist_pcatsne', linsep)
