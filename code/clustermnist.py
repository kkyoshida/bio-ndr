import numpy as np
import matplotlib.pyplot as plt
import figplot
import model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

np.random.seed(2)

def cluster_kmeans(X, n_clu, r):
    clustering = KMeans(n_clusters=n_clu, random_state=r, n_init=10).fit_predict(X)
    return clustering

def calculate_randscore(X, n_clu, label):
    ans = 0.0
    trial_r = 10 # averaged by multiple trials
    for r in range(trial_r):
        cluster_index = cluster_kmeans(X, n_clu, r)
        ans += adjusted_rand_score(cluster_index, label)
    return ans / trial_r

mnist, mnist_label = model.data_load(data_index=2)

point_size = 0.1
set_width = 0.5

# Linear separability and Clustering metrics
trial_number = 5

linsep_mnist_som = np.zeros(trial_number)
clustering_mnist_som = np.zeros(trial_number)

# SOM
mnist_reduced_som_list = np.load('../data/som_result_mnist.npy')
for j in range(trial_number):
    linsep_mnist_som[j] = 100 * model.lin_separate(mnist_reduced_som_list[j], mnist_label)
    clustering_mnist_som[j] = calculate_randscore(mnist_reduced_som_list[j], 10, mnist_label)

# K-means
clustering_mnist_kmeans = calculate_randscore(mnist, 10, mnist_label)
  
print(np.mean(linsep_mnist_som))

# PCA
pca_mnist = PCA(n_components=2, svd_solver='full', random_state=1)
pca_mnist.fit(mnist)
Xd_mnist = pca_mnist.transform(mnist)
linsep_mnist_pca = 100 * model.lin_separate(Xd_mnist, mnist_label)
clustering_mnist_pca = calculate_randscore(Xd_mnist, 10, mnist_label)

# t-SNE and Hebbian t-SNE
linsep_mnist_tsne_40 = np.zeros(trial_number)
linsep_mnist_hebbtsne = np.zeros(trial_number)
clustering_mnist_tsne_40 = np.zeros(trial_number)
clustering_mnist_hebbtsne = np.zeros(trial_number)

for j in range(trial_number):
    # t-SNE
    tsne = TSNE(n_components=2, random_state=j, perplexity=40)
    mnist_reduced = tsne.fit_transform(mnist)
    linsep_mnist_tsne_40[j] = 100 * model.lin_separate(mnist_reduced, mnist_label)
    clustering_mnist_tsne_40[j] = calculate_randscore(mnist_reduced, 10, mnist_label)
    # Hebbian t-SNE
    npz_comp = np.load('../data/mnist40-' + '{}.npz'.format(j))
    mnist_reduced_hebbtsne = npz_comp['arr_0']
    linsep_mnist_hebbtsne[j] = 100 * model.lin_separate(mnist_reduced_hebbtsne, mnist_label)
    clustering_mnist_hebbtsne[j] = calculate_randscore(mnist_reduced_hebbtsne, 10, mnist_label)

print(clustering_mnist_pca, np.mean(clustering_mnist_tsne_40), np.mean(clustering_mnist_hebbtsne))
print(linsep_mnist_pca, np.mean(linsep_mnist_tsne_40), np.mean(linsep_mnist_hebbtsne))

linsep_mnist_pcatsne = np.load('../data/linsep_mnist_pcatsne.npy')
print(linsep_mnist_pcatsne[0], linsep_mnist_pcatsne[1])

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(4, 4, 1)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
ax.bar([0, 1, 2, 3], [linsep_mnist_pca, np.mean(linsep_mnist_tsne_40), np.mean(linsep_mnist_hebbtsne), np.mean(linsep_mnist_som)]\
       , yerr=[np.nan, np.std(linsep_mnist_tsne_40, ddof=1)/np.sqrt(trial_number), np.std(linsep_mnist_hebbtsne, ddof=1)/np.sqrt(trial_number), \
               np.std(linsep_mnist_som, ddof=1)/np.sqrt(trial_number)], \
                tick_label=['PCA', 't-SNE', 'Hebbian\nt-SNE', 'SOM'], capsize=3, color=['red', 'orange', 'blue', 'purple'], width=0.7)
ax.set_aspect(0.04)
ax.set_ylim(0, 80) 

ax = fig.add_subplot(4, 4, 2)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
ax.bar([0, 1, 2, 3, 4], [clustering_mnist_pca, np.mean(clustering_mnist_tsne_40), np.mean(clustering_mnist_hebbtsne), \
                            np.mean(clustering_mnist_som), clustering_mnist_kmeans], \
       yerr=[np.nan, np.std(clustering_mnist_tsne_40, ddof=1)/np.sqrt(trial_number), np.std(clustering_mnist_hebbtsne, ddof=1)/np.sqrt(trial_number), \
            np.std(clustering_mnist_som, ddof=1)/np.sqrt(trial_number), np.nan], \
       tick_label=['PCA', 't-SNE', 'Hebbian\nt-SNE', 'SOM', 'K-means'], capsize=3, color=['red', 'orange', 'blue', 'purple', '#ADD8E6'], width=0.7)
ax.set_aspect(6.25)
ax.set_ylim(0, 0.5) 

plt.savefig('../figure/mnistclusteringmetric.pdf')

# plot SOM and K-means
fig = plt.figure(figsize=(8, 8))

# SOM
ax = fig.add_subplot(4, 4, 1)
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
mnist_reduced_som_sample = mnist_reduced_som_list[0, :] + 0.01 * np.array([[i%40, int(i/40)] for i in range(np.shape(mnist_reduced_som_list[0, :])[0])])
plot_mnist_som = ax.scatter(mnist_reduced_som_sample[:, 0], mnist_reduced_som_sample[:, 1], c=mnist_label, cmap='jet', s=0.5, linewidths=0)

# K-means
cluster_size = 10
clustering = KMeans(n_clusters=cluster_size, random_state=0, n_init=10).fit_predict(mnist)

# Plot K-mean
def make_cross_matrix(matrix_size, true_label, cluster_label):
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    for true, cluster in zip(true_label, cluster_label):
        confusion_matrix[true, cluster] += 1
    return confusion_matrix

def result_matrix(matrix_size, true_label, cluster_label):
    row_ind, col_ind = linear_sum_assignment(-make_cross_matrix(matrix_size=matrix_size, true_label=true_label, cluster_label=cluster_label))
    cluster_labels_changed = np.zeros(np.size(cluster_label), dtype=int)
    for i in range(np.size(true_label)):
        cluster_labels_changed[i] = np.where(col_ind == cluster_label[i])[0][0]
    result_matrix = make_cross_matrix(matrix_size=matrix_size, true_label=true_label, cluster_label=cluster_labels_changed)
    return result_matrix

result_kmeans = result_matrix(matrix_size=10, true_label=mnist_label.astype(int), cluster_label=clustering)

ax = fig.add_subplot(4, 4, 2)
cax = ax.imshow(result_kmeans, cmap='Blues', interpolation='nearest')
ax.set_ylabel("True Labels")
ax.spines['right'].set_linewidth(set_width)
ax.spines['left'].set_linewidth(set_width)
ax.spines['top'].set_linewidth(set_width)
ax.spines['bottom'].set_linewidth(set_width)
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.xaxis.set_ticks_position('top')  
ax.tick_params(top=True, bottom=False)  

for i in range(10):
    for j in range(10):
        ax.text(j, i, result_kmeans[i, j], ha='center', va='center', color='black')

plt.savefig('../figure/othermethods.pdf')
