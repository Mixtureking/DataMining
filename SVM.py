from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


mean_x1, mean_y1 = np.random.uniform(2, 3), np.random.uniform(2, 3)
mean_x2, mean_y2 = np.random.uniform(4, 5), np.random.uniform(4, 5)
means = [[mean_x1, mean_y1], [mean_x2, mean_y2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1
X = np.concatenate((X0.T, X1.T), axis = 1) # all data
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels

# Hiển thị số lượng
plt.plot(X0[:,0], X0[:, 1], 'bs', markersize = 8, alpha = .8)
plt.plot(X1[:,0], X1[:, 1], 'ro', markersize = 8, alpha = .8)
plt.axis('equal')
plt.ylim(0,6)
plt.xlim(0,8)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlabel('$x_1$',fontsize = 20)
plt.ylabel('$x_2$',fontsize = 20)

plt.show()

from cvxopt import matrix, solvers
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((2*N, 1))) # all-one vector
# build A, b, G, h
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1)))
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)

# Vẽ dữ liệu
plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='blue', marker='s', label="Class 1")
plt.scatter(X1[:, 0], X1[:, 1], c='red', marker='o', label="Class -1")
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.legend()
plt.axis("equal")

# Vẽ đường biên SVM
x_min, x_max = 0, 8
x_vals = np.linspace(x_min, x_max, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'k-', linewidth=2, label='Decision Boundary')

# Vẽ đường lề (margin)
margin = 1 / np.linalg.norm(w)
y_vals_plus = y_vals + margin
y_vals_minus = y_vals - margin
plt.plot(x_vals, y_vals_plus, 'k--', linewidth=1)
plt.plot(x_vals, y_vals_minus, 'k--', linewidth=1)

# Đánh dấu support vectors
plt.scatter(XS[0, :], XS[1, :], s=100, facecolors='none', edgecolors='k', label="Support Vectors")

plt.ylim(0, 6)
plt.xlim(0, 8)
plt.legend()
plt.show()

