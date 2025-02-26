import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(2)

# Tạo dữ liệu
means = [[2, 2], [4, 2]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.hstack((X0, X1))
y = np.hstack((np.ones(N), -np.ones(N)))

# Thêm bias vào X
X = np.vstack((np.ones(2 * N), X))


# Hàm tính dự đoán
def predict(w, x):
    return np.sign(np.dot(w.T, x))


# Kiểm tra hội tụ
def has_converged(X, y, w):
    return np.all(predict(w, X) == y)


# Thuật toán Perceptron
def perceptron(X, y, w_init):
    w = [w_init]
    mis_points = []
    N = X.shape[1]

    while True:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(-1, 1)
            yi = y[i]
            if predict(w[-1], xi)[0] != yi:
                mis_points.append(i)
                w.append(w[-1] + yi * xi)

        if has_converged(X, y, w[-1]):
            break
    return w, mis_points


# Trực quan hóa đường phân chia
def draw_line(w):
    w0, w1, w2 = w.flatten()
    x_vals = np.array([0, 6])
    y_vals = -(w1 * x_vals + w0) / w2
    plt.plot(x_vals, y_vals, 'k')


# Khởi tạo trọng số và huấn luyện PLA
w_init = np.random.randn(3, 1)
w, misclassified = perceptron(X, y, w_init)


# Trực quan hóa thuật toán Perceptron
def visualize_perceptron(w, misclassified):
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(i):
        plt.cla()
        plt.scatter(X0[0], X0[1], marker='^', color='blue', label="Class 1", alpha=0.8)
        plt.scatter(X1[0], X1[1], marker='o', color='red', label="Class -1", alpha=0.8)
        draw_line(w[min(i, len(w) - 1)])

        if i < len(misclassified):
            circle = plt.Circle((X[1, misclassified[i]], X[2, misclassified[i]]), 0.15, color='k', fill=False)
            ax.add_artist(circle)

        ax.set_xlim(0, 6)
        ax.set_ylim(-2, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'PLA: Iter {min(i, len(w) - 1)}/{len(w) - 1}')
        plt.legend()

    anim = FuncAnimation(fig, update, frames=len(w) + 1, interval=1000)
    anim.save('pla_vis.gif', dpi=100, writer='imagemagick')
    plt.show()


visualize_perceptron(w, misclassified)