import numpy as np
import matplotlib.pyplot as plt
import figplot
import model
import params
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

set_width = 0.5

np.random.seed(2)

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

# Hallem2006
data, ylabel = model.data_load(data_index=4)
data_kc_sparseness = np.load('../data/kc_hallem_sparseness.npz')['arr_0']

for perp_set in range(2):
    fig = plt.figure(figsize=(8, 8))
    trial_number = params.trial_test
    spar = np.where(params.sparseness_list == 0.05)[0][0] # in the case of 5% for Hebbian t-SNE and KC PCA

    if perp_set == 0:
        # PCA
        pca = PCA(n_components=2, svd_solver='full', random_state=1)
        pca.fit(data)
        Xd = pca.transform(data)

        figplot.plot_points(fig, Xd, ylabel, 4, 4, 1)
        linsep_pca = 100 * model.lin_separate(Xd, ylabel)
        clustering_pca = calculate_randscore(X=Xd, n_clu=10, label=ylabel)

        # PCA of KCs
        linsep_kc = np.zeros(trial_number)
        clustering_kc = np.zeros(trial_number)
        for rand_state in range(trial_number):
            data_kc = data_kc_sparseness[spar, rand_state].T
            pcakc = PCA(n_components=2, svd_solver='full', random_state=1)
            pcakc.fit(data_kc)
            Xd_kc = pcakc.transform(data_kc)
            linsep_kc[rand_state] = 100 * model.lin_separate(Xd_kc, ylabel)
            clustering_kc[rand_state] = calculate_randscore(X=Xd_kc, n_clu=10, label=ylabel)

        # K-means
        clustering_kmeans = np.zeros(trial_number)
        kmeans_clustering_result_list = np.zeros((trial_number, np.size(ylabel)), dtype=int)
        for trial in range(trial_number):
            kmeans_clustering_result_list[trial] = KMeans(n_clusters=10, random_state=trial, n_init=10).fit_predict(data)   
            clustering_kmeans[trial] = adjusted_rand_score(kmeans_clustering_result_list[trial], ylabel)

        plot_example_kmeans = np.argsort(clustering_kmeans)[int(trial_number/2)]

        # SOM
        linsep_som = np.zeros(trial_number)
        clustering_som = np.zeros(trial_number)
        som_reduced_list = np.load('../data/som_result_hallem.npy')
        for trial in range(trial_number):
            som_reduced = som_reduced_list[trial]
            linsep_som[trial] = 100 * model.lin_separate(som_reduced, ylabel)
            clustering_som[trial] = calculate_randscore(X=som_reduced, n_clu=10, label=ylabel)

        plot_example_som = np.argsort(linsep_som)[int(trial_number/2)]
        som_reduced = som_reduced_list[plot_example_som]

        som_reduced += 0.03 * np.array([[i%11, int(i/11)] for i in range(np.shape(som_reduced)[0])]) 

        plot_points_overlap(fig, som_reduced, ylabel, 4, 4, 4)

    # Hebbian t-SNE
    linsep_tsne = np.zeros(trial_number)
    clustering_tsne = np.zeros(trial_number)

    for rand_state in range(trial_number):
        w_hebb = np.load('../data/hallemtsne-sparse{}-{}.npz'.format([20, 30][perp_set], 100*spar+rand_state))['arr_0']
        linsep_tsne[rand_state] = 100 * model.lin_separate(w_hebb, ylabel)
        clustering_tsne[rand_state] = calculate_randscore(X=w_hebb, n_clu=10, label=ylabel)

    plot_example_hebbtsne = np.argsort(linsep_tsne)[int(trial_number/2)]
    w_hebb = np.load('../data/hallemtsne-sparse{}-{}.npz'.format([20, 30][perp_set], 100*spar+plot_example_hebbtsne))['arr_0']
    figplot.plot_points(fig, w_hebb, ylabel, 4, 4, 3)
    
    trial_number_original = 100
    linsep_tsne_original = np.zeros(trial_number_original)
    clustering_tsne_original = np.zeros(trial_number_original)
    for rand_state in range(trial_number_original):
        tsne_original = TSNE(n_components=2, random_state=rand_state, perplexity=[20, 30][perp_set])
        tsne_original_reduced = tsne_original.fit_transform(data)
        linsep_tsne_original[rand_state] = 100 * model.lin_separate(tsne_original_reduced, ylabel)
        clustering_tsne_original[rand_state] = calculate_randscore(X=tsne_original_reduced, n_clu=10, label=ylabel)
    plot_example_tsne = np.argsort(linsep_tsne_original)[int(trial_number_original/2)]
    tsne_original = TSNE(n_components=2, random_state=plot_example_tsne, perplexity=[20, 30][perp_set])
    tsne_original_reduced = tsne_original.fit_transform(data)
    figplot.plot_points(fig, tsne_original_reduced, ylabel, 4, 4, 2)
    
    ax = fig.add_subplot(4, 4, 5)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.bar([0, 1, 2], [linsep_pca, np.mean(linsep_tsne_original), np.mean(linsep_tsne)], yerr=[np.nan, np.std(linsep_tsne_original, ddof=1)/np.sqrt(trial_number_original), np.std(linsep_tsne, ddof=1)/np.sqrt(trial_number)], tick_label=['PCA', 't-SNE', 'Hebbian\nt-SNE'], capsize=3, color=['red', 'orange', 'blue'], width=0.7)
    ax.set_aspect(0.05)
    ax.set_ylim(0, 63) 
    ax.axhline(np.mean(linsep_kc), c='green', linestyle='dashed')

    ax = fig.add_subplot(4, 4, 6)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)
    ax.bar([0, 1], [np.mean(linsep_tsne), np.mean(linsep_som)], \
        yerr=[np.std(linsep_tsne, ddof=1)/np.sqrt(trial_number), \
                np.std(linsep_som, ddof=1)/np.sqrt(trial_number)], tick_label=['Hebbian\nt-SNE', 'SOM'], \
                    capsize=3, color=['blue', 'purple'], width=0.7)
    ax.set_aspect(0.05)
    ax.set_ylim(0, 63) 
    plt.savefig('../figure/Hallem2006-{}-r.pdf'.format([20, 30][perp_set]))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(4, 2, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)

    if perp_set==0:
        ax.bar([0, 1, 2, 3, 4], [clustering_pca, np.mean(clustering_tsne_original), np.mean(clustering_tsne), np.mean(clustering_som), np.mean(clustering_kmeans)], \
            yerr=[np.nan, np.std(clustering_tsne_original, ddof=1)/np.sqrt(trial_number_original), np.std(clustering_tsne, ddof=1)/np.sqrt(trial_number), np.std(clustering_som, ddof=1)/np.sqrt(trial_number), \
                    np.nan], tick_label=['PCA', 't-SNE', 'Hebbian\nt-SNE', 'SOM', 'K-means'], \
                        capsize=3, color=['red', 'orange', 'blue', 'purple', '#ADD8E6'], width=0.7)
    elif perp_set==1:
        ax.bar([0, 1, 2], [clustering_pca, np.mean(clustering_tsne_original), np.mean(clustering_tsne)], \
            yerr=[np.nan, np.std(clustering_tsne_original, ddof=1)/np.sqrt(trial_number_original), np.std(clustering_tsne, ddof=1)/np.sqrt(trial_number)], \
                    tick_label=['PCA', 't-SNE', 'Hebbian\nt-SNE'], \
                        capsize=3, color=['red', 'orange', 'blue'], width=0.7)
    ax.set_aspect(10)
    ax.set_ylim(0, 0.315) 
    ax.axhline(np.mean(clustering_kc), c='green', linestyle='dashed')
   

    plt.savefig('../figure/Hallem2006-clustering-{}-r.pdf'.format([20, 30][perp_set]))

    print(perp_set, linsep_pca, np.mean(linsep_tsne), np.mean(linsep_tsne_original))
    print(perp_set, clustering_pca, np.mean(clustering_tsne), np.mean(clustering_tsne_original))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(-1.0, 1.0)
    sc = ax.scatter(np.array([int(i/2) for i in range(10)]), np.array([- (i%2) * 0.1 for i in range(10)]), c=np.array([i for i in range(10)]), cmap='jet', s=10)
    plt.colorbar(sc)
    plt.savefig('../figure/Hallemcolor-{}-r.pdf'.format([20, 30][perp_set]))

    #sparse pattern
    linsep_tsne_sparse = np.zeros((np.shape(data_kc_sparseness)[0], np.shape(data_kc_sparseness)[1]))
    linsep_kcpca_sparse = np.zeros((np.shape(data_kc_sparseness)[0], np.shape(data_kc_sparseness)[1]))
    for i in range(np.shape(data_kc_sparseness)[0]):
        for trial in range(np.shape(data_kc_sparseness)[1]):
            w_hebb = np.load('../data/hallemtsne-sparse{}-{}.npz'.format([20, 30][perp_set], 100*i+trial))['arr_0']
            linsep_tsne_sparse[i, trial] = 100 * model.lin_separate(w_hebb, ylabel)

            data_kc = data_kc_sparseness[i, trial].T
            pcakc = PCA(n_components=2, svd_solver='full', random_state=1)
            pcakc.fit(data_kc)
            kc_repre_pca = pcakc.transform(data_kc)
            linsep_kcpca_sparse[i, trial] = 100 * model.lin_separate(kc_repre_pca, ylabel)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(4, 4, 1)
    ax.spines['right'].set_linewidth(set_width)
    ax.spines['left'].set_linewidth(set_width)
    ax.spines['top'].set_linewidth(set_width)
    ax.spines['bottom'].set_linewidth(set_width)

    ax.axhline(linsep_pca, c='red')
    ax.axhline(np.mean(linsep_tsne_original), c='orange')

    ans_mean_tsne = np.mean(linsep_tsne_sparse, axis=1)
    ans_se_tsne = np.std(linsep_tsne_sparse, axis=1, ddof=1) / np.sqrt(np.shape(linsep_tsne_sparse)[1])
    ax.errorbar(params.sparseness_list, ans_mean_tsne, yerr=ans_se_tsne, capsize=2, color='blue', marker='.', markersize=2)

    ans_mean_kcpca = np.mean(linsep_kcpca_sparse, axis=1)
    ans_se_kcpca = np.std(linsep_kcpca_sparse, axis=1, ddof=1) / np.sqrt(np.shape(linsep_kcpca_sparse)[1])
    ax.errorbar(params.sparseness_list, ans_mean_kcpca, yerr=ans_se_kcpca, capsize=2, color='green', marker='.', markersize=2, fmt='--')
    ax.set_aspect(0.01) 
    ax.set_ylim(0, 63)
    plt.savefig('../figure/Hallem-sparseness-{}-r.pdf'.format([20, 30][perp_set]))

   
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

result_kmeans = result_matrix(matrix_size=10, true_label=ylabel.astype(int), cluster_label=kmeans_clustering_result_list[plot_example_kmeans])
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(4, 4, 1)

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

plt.savefig('../figure/Hallem2006-Kmeans-plot.pdf')

