from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

# Fake du lieu
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3
original_label = np.asarray([0] * N + [1] * N + [2] * N).T


# ham hien thi du lieu tren do thi
def kmeans_display(X, label):
#     K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

    
# khoi tao cac gia tri ban dau, ngau nhien chon ra K diem lam center
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]


# gan nhan cho cac diem la trung tam co khoang cach ngan nhat
def kmeans_assign_labels(X, centers):
    # tinh khoang cach cua tat ca cac diem trong tap X voi tat ca cac center
    D = cdist(X, centers)
    # gan nhan la trung tam gan nhat
    return np.argmin(D, axis=1)


# dua tren cac nhan vua duoc gan, cap nhat lai center moi cho k nhom
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # lay tat ca cac diem thuoc tung nhom 
        Xk = X[labels == k, :]
        # lay gia tri trung binh lam center
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


# kiem tra dieu kien dung, neu sau khi set lai center, tat ca cac diem khong thay doi center truoc khi set
def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

    
# thuat toan kmeans
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    print('First centers:')
    print(centers[-1])
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

    
(centers, labels, it) = kmeans(X, K)
print('Centers found by algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
