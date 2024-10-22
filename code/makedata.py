import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_s_curve

t1 = time.time() 

np.random.seed(1)

#data of rings
def make_rings(points_per_ring, n_rings):
    data = np.zeros((n_rings * points_per_ring, 3))
    if n_rings == 2:
        for i in range(points_per_ring):
            data[i] = 1000 * np.array([np.sin(i * 2 * np.pi / points_per_ring), np.cos(i * 2 * np.pi / points_per_ring), 0])
            data[i + points_per_ring] = 1000 * np.array([np.sin(i * 2 * np.pi / points_per_ring) + 1, 0, np.cos(i * 2 * np.pi / points_per_ring)])
    if n_rings == 4:
        for j in range(n_rings):
            for i in range(points_per_ring):
                if j%2 == 0:
                    data[i + j * points_per_ring] = 1000 * np.array([np.sin(i * 2 * np.pi / points_per_ring) + 4 / 3 * j, np.cos(i * 2 * np.pi / points_per_ring), 0])
                else:
                    data[i + j * points_per_ring] = 1000 * np.array([np.sin(i * 2 * np.pi / points_per_ring) + 4 / 3 * j, 0, np.cos(i * 2 * np.pi / points_per_ring)])
    ylabel = np.array([int(i/points_per_ring) for i in range(n_rings * points_per_ring)])
    return data, ylabel

#MNIST
def make_mnist(n_points):
    data, ylabel = fetch_openml('mnist_784', as_frame=False, return_X_y=True, version=1)
    ylabel = ylabel.astype(int)
    rand_permutation = np.random.permutation(np.size(ylabel))
    idx = rand_permutation[0 : n_points] # randomly select n_points data
    return data[idx], ylabel[idx]

#data of S-curve
def make_s_shaped_curve():
    sshape, sshape_label = make_s_curve(n_samples=400, noise=0.0, random_state=0)
    sshape = 100 * sshape
    index = np.argsort(sshape_label)
    sshape = sshape[index]
    sshape_label = sshape_label[index]
    sshape_label[0 : int(0.5 * np.size(sshape_label))] = 0
    sshape_label[int(0.5 * np.size(sshape_label)) : np.size(sshape_label)] = 1
    sshape[:, 0] *= 1.5
    return sshape, sshape_label

ring1, ring1_label = make_rings(100, 2)
ring2, ring2_label = make_rings(100, 4)
mnist, mnist_label = make_mnist(1200)
sshape, sshape_label = make_s_shaped_curve()

np.save('../data/ring1', ring1)
np.save('../data/ring1_label', ring1_label)
np.save('../data/ring2', ring2)
np.save('../data/ring2_label', ring2_label)
np.save('../data/mnist', mnist)
np.save('../data/mnist_label', mnist_label)
np.save('../data/sshape', sshape)
np.save('../data/sshape_label', sshape_label)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")