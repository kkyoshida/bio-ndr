import numpy as np
import matplotlib.pyplot as plt
import model
import figplot
import params
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

set_width = 0.5

np.random.seed(2)

def cluster_kmeans(X, n_clu, r):
    clustering = KMeans(n_clusters=n_clu, random_state=r, n_init=10).fit_predict(X)
    return clustering

def calculate_variance(X, n_clu, label, trial_r=10):
    ans = 0.0
    # averaged by multiple trials
    for r in range(trial_r):
        cluster_index = cluster_kmeans(X, n_clu, r)
        for j in range(n_clu):
            ans += np.var(label[cluster_index == j]) * np.size(label[cluster_index == j])
    return ans / (trial_r * np.size(label))

def distance_list(X, repre_dim):
    distances = np.sqrt(np.sum((X.reshape(-1, 1, repre_dim) - X.reshape(1, -1, repre_dim))**2, axis=2))
    distances = distances[np.triu_indices_from(distances, k=1)]
    return distances/np.max(distances)


data, label = model.data_load(data_index=5)
trial_number = params.trial_test

# determine the best cluster number in K-means
score = np.zeros((trial_number, np.size(label)-2))
for k in range(2, np.size(label)):
    for trial in range(trial_number):
        cluster_labels = KMeans(n_clusters=k, random_state=100+trial, n_init=10).fit_predict(data)
        score[trial, k-2] = silhouette_score(data, cluster_labels)
cluster_n_kmeans = np.argmax(np.mean(score, axis=0)) + 2
print(cluster_n_kmeans)

# Plot
for perp_set in range(2):
    # PCA
    pca = PCA(n_components=2, svd_solver='full', random_state=1)
    pca.fit(data)
    Xd = pca.transform(data)

    # Hebbian t-SNE
    Xd_hebbtsne = np.load('../data/badeltsne{}-0.npz'.format([20, 30][perp_set]))['arr_0']

    # t-SNE
    tsne_original = TSNE(n_components=2, random_state=0, perplexity=[20, 30][perp_set])
    Xd_originaltsne = tsne_original.fit_transform(data)
    
    fig = plt.figure(figsize=(4, 4))
    figplot.plot_points(fig, Xd, label, 2, 2, 1)
    figplot.plot_points(fig, Xd_hebbtsne, label, 2, 2, 2)
    figplot.plot_points(fig, Xd_originaltsne, label, 2, 2, 3)
    ax = fig.add_subplot(2, 2, 4)
    img = ax.scatter(Xd[:,0], Xd[:,1], c=label, cmap='jet', s=0.8)
    fig.colorbar(img)

    plt.tight_layout()
    plt.savefig('../figure/badelsampleplot{}-r.pdf'.format([20, 30][perp_set]), dpi=300)

    dis_pca = distance_list(Xd, repre_dim=2)
    dis_hebbtsne = distance_list(Xd_hebbtsne, repre_dim=2)
    dis_originaltsne = distance_list(Xd_originaltsne, repre_dim=2)
    label_dis = distance_list(label, repre_dim=1)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(2, 2, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.scatter(label_dis, dis_pca, s=0.1)
    corr_coefficient, p_value = pearsonr(label_dis, dis_pca, alternative='greater')
    ax.text(0.05, 0.95, f'Pearson r: {corr_coefficient:.2f}\nP-value: {p_value:.2e}', transform=ax.transAxes, verticalalignment='top')

    ax = fig.add_subplot(2, 2, 2)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.scatter(label_dis, dis_hebbtsne, s=0.1)
    corr_coefficient, p_value = pearsonr(label_dis, dis_hebbtsne, alternative='greater')
    ax.text(0.05, 0.95, f'Pearson r: {corr_coefficient:.2f}\nP-value: {p_value:.2e}', transform=ax.transAxes, verticalalignment='top')

    ax = fig.add_subplot(2, 2, 3)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.scatter(label_dis, dis_originaltsne, s=0.1)
    corr_coefficient, p_value = pearsonr(label_dis, dis_originaltsne, alternative='greater')
    ax.text(0.05, 0.95, f'Pearson r: {corr_coefficient:.2f}\nP-value: {p_value:.2e}', transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../figure/badeldistanceplot{}-r.pdf'.format([20, 30][perp_set]))

# Clustering analysis
result_highdinput = np.zeros(np.size(label))
result_pca = np.zeros(np.size(label))
result_kcpca = np.zeros((np.size(label), trial_number))
result_kmeans = np.zeros((np.size(label), trial_number))
result_som = np.zeros((np.size(label), trial_number))

for i in range(np.size(label)):
    n_clu = i + 1    
    for j in range(trial_number):
        # PCA of KC activities
        data_kc = np.load('../data/kc_badel.npy')[j].T
        pcakc = PCA(n_components=2, svd_solver='full', random_state=1)
        pcakc.fit(data_kc)
        Xd_kc = pcakc.transform(data_kc)
        result_kcpca[i, j] = calculate_variance(Xd_kc, n_clu, label)

        # K-means
        clustering = KMeans(n_clusters=cluster_n_kmeans, random_state=j, n_init=10).fit_predict(data)
        rand_position = np.random.normal(loc=0, scale=1, size=(cluster_n_kmeans, 2)) 
        
        min_distance = np.min(np.array([[np.sqrt(np.sum((rand_position[k] - rand_position[l]) ** 2)) if k != l else np.inf for l in range(rand_position.shape[0])] for k in range(rand_position.shape[0])]))
        if min_distance == 0:
            print('error')
        rand_position /= min_distance # the minimum distance is set to one
        kmeans_reduced = rand_position[clustering]  
        kmeans_reduced += np.random.uniform(0, 0.01, np.shape(kmeans_reduced)) 
        result_kmeans[i, j] = calculate_variance(kmeans_reduced, n_clu, label)

        # SOM
        som_reduced_list = np.load('../data/som_result_badel.npy')
        for j in range(trial_number):
            som_reduced = som_reduced_list[j]
            som_reduced += np.random.uniform(0, 0.01, np.shape(som_reduced)) 
            result_som[i, j] = calculate_variance(som_reduced, n_clu, label)

    result_pca[i] = calculate_variance(Xd, n_clu, label)
    result_highdinput[i] = calculate_variance(data, n_clu, label, trial_r=100)

result_pca -= result_highdinput
result_kcpca -= result_highdinput.reshape(-1, 1)
result_kmeans -= result_highdinput.reshape(-1, 1)
result_som -= result_highdinput.reshape(-1, 1)

for perp_set in range(2):
    result_hebbtsne = np.zeros((np.size(label), trial_number))
    result_originaltsne = np.zeros((np.size(label), trial_number))
    result_rand = np.zeros((np.size(label), trial_number))

    # t-SNE
    Xd_originaltsne_list = np.zeros((trial_number, np.size(label), 2))
    for j in range(trial_number):
        tsne_original = TSNE(n_components=2, random_state=j, perplexity=[20, 30][perp_set])
        Xd_originaltsne_list[j] = tsne_original.fit_transform(data)
    
    for i in range(np.size(label)):
        n_clu = i + 1    
        for j in range(trial_number):
            # Hebbina t-SNE
            w_hebb = np.load('../data/badeltsne{}-{}.npz'.format([20, 30][perp_set], j))['arr_0']
            result_hebbtsne[i, j] = calculate_variance(w_hebb, n_clu, label)

            # original t-SNE
            result_originaltsne[i, j] = calculate_variance(Xd_originaltsne_list[j], n_clu, label)

            # label shuffled
            rand_permutation = np.random.permutation(np.size(label))
            result_rand[i, j] = calculate_variance(w_hebb, n_clu, label[rand_permutation])
        
    fig = plt.figure(figsize=(3, 6))
    x_index = np.array([(i+1) for i in range(np.size(label))])

    # difference from high-dimensional input
    result_hebbtsne -= result_highdinput.reshape(-1, 1)
    result_originaltsne -= result_highdinput.reshape(-1, 1)
    result_rand -= result_highdinput.reshape(-1, 1)

    for j in range(2):
        ax = fig.add_subplot(4, 1, 1 + j)
        ax.plot(x_index, result_pca, c='red', linestyle='dashed', label='PCA')
        ax.plot(x_index, np.mean(result_kcpca, axis=1), c='green', linestyle='dashdot', label='PCA of KCs')
        ax.plot(x_index, np.mean(result_hebbtsne, axis=1), c='blue', linestyle='solid', label='Hebbian t-SNE')
        ax.plot(x_index, np.mean(result_originaltsne, axis=1), c='orange', linestyle=(0, (1, 1)), label='t-SNE')
        ax.plot(x_index, np.mean(result_rand, axis=1), c='gray', linestyle='dotted', label='Shuffled')
        ax.fill_between(x_index, np.mean(result_kcpca, axis=1) + np.std(result_kcpca, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_kcpca, axis=1) - np.std(result_kcpca, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='green')
        ax.fill_between(x_index, np.mean(result_hebbtsne, axis=1) + np.std(result_hebbtsne, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_hebbtsne, axis=1) - np.std(result_hebbtsne, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='blue')
        ax.fill_between(x_index, np.mean(result_originaltsne, axis=1) + np.std(result_originaltsne, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_originaltsne, axis=1) - np.std(result_originaltsne, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='orange')
        ax.fill_between(x_index, np.mean(result_rand, axis=1) + np.std(result_rand, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_rand, axis=1) - np.std(result_rand, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='gray')
        ax.set_ylim(-0.2/84, 0.8/84)

        if j==1:
            ax.legend()

    for j in range(2):
        ax = fig.add_subplot(4, 1, 3 + j)
        ax.plot(x_index, np.mean(result_kmeans, axis=1), c='#ADD8E6', linestyle='dashed', label='K-means')
        ax.plot(x_index, np.mean(result_som, axis=1), c='purple', linestyle='dashdot', label='SOM')
        ax.plot(x_index, np.mean(result_hebbtsne, axis=1), c='blue', linestyle='solid', label='Hebbian t-SNE')
        ax.plot(x_index, np.mean(result_rand, axis=1), c='gray', linestyle='dotted', label='Shuffled')
        ax.fill_between(x_index, np.mean(result_kmeans, axis=1) + np.std(result_kmeans, axis=1, ddof=1)/np.sqrt(trial_number), \
                        np.mean(result_kmeans, axis=1) - np.std(result_kmeans, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='#ADD8E6')
        ax.fill_between(x_index, np.mean(result_som, axis=1) + np.std(result_som, axis=1, ddof=1)/np.sqrt(trial_number), \
                        np.mean(result_som, axis=1) - np.std(result_som, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='purple')
        ax.fill_between(x_index, np.mean(result_hebbtsne, axis=1) + np.std(result_hebbtsne, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_hebbtsne, axis=1) - np.std(result_hebbtsne, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='blue')
        ax.fill_between(x_index, np.mean(result_rand, axis=1) + np.std(result_rand, axis=1, ddof=1)/np.sqrt(trial_number), np.mean(result_rand, axis=1) - np.std(result_rand, axis=1, ddof=1)/np.sqrt(trial_number), alpha=0.15, color='gray')
        ax.set_ylim(-0.2/84, 0.8/84)

        if j==1:
            ax.legend()
    
    plt.savefig('../figure/badelcluster{}-re.pdf'.format([20, 30][perp_set]))


def plot_points_overlap(fig, X_f, y_label_f, p_x, p_y, p_n):
    ax = fig.add_subplot(p_x, p_y, p_n)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    plot_xmin = np.min(X_f[:, 0]) - 0.05 * (np.max(X_f[:, 0]) - np.min(X_f[:, 0]))
    plot_xmax = np.max(X_f[:, 0]) + 0.05 * (np.max(X_f[:, 0]) - np.min(X_f[:, 0]))
    plot_ymin = np.min(X_f[:, 1]) - 0.05 * (np.max(X_f[:, 1]) - np.min(X_f[:, 1]))
    plot_ymax = np.max(X_f[:, 1]) + 0.05 * (np.max(X_f[:, 1]) - np.min(X_f[:, 1]))
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)
    ax.scatter(X_f[:,0], X_f[:,1], c=y_label_f, cmap='jet', s=2.0, linewidths=0)
    ax.set_aspect(1 / (plot_ymax-plot_ymin) * (plot_xmax-plot_xmin))

fig = plt.figure(figsize=(4, 4))
som_reduced = np.load('../data/som_result_badel.npy')[0]
som_reduced += 0.03 * np.array([[i%10, int(i/10)] for i in range(np.shape(som_reduced)[0])]) 
plot_points_overlap(fig, som_reduced, label, 2, 2, 1)

plt.tight_layout()
plt.savefig('../figure/badelSOM_sampleplot.pdf')
