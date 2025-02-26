import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.svm import SVC

# Đặt seed để kết quả nhất quán
np.random.seed(21)

# Tạo dữ liệu cho hai lớp
means = [[2, 2], [4, 1]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

# Gộp dữ liệu và tạo nhãn
X = np.vstack((X0, X1))
Y = np.hstack((np.ones(N), -np.ones(N)))

# Vẽ và lưu biểu đồ dữ liệu ban đầu
with PdfPages('data.pdf') as pdf:
    plt.figure(figsize=(5, 5))
    plt.scatter(X0[:, 0], X0[:, 1], color='blue', marker='s', label="Class 1", alpha=0.8)
    plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='o', label="Class -1", alpha=0.8)

    plt.xlim(0, 5)
    plt.ylim(0, 4)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.legend()

    pdf.savefig()
    plt.show()

# Huấn luyện SVM với kernel tuyến tính
C = 100
clf = SVC(kernel='linear', C=C)
clf.fit(X, Y)

# Lấy vector trọng số và bias
w_sklearn = clf.coef_.flatten()
b_sklearn = clf.intercept_[0]

# Vẽ và lưu biểu đồ SVM
with PdfPages('SKlearn.pdf') as pdf:
    plt.figure(figsize=(5, 5))

    # Vẽ dữ liệu
    plt.scatter(X0[:, 0], X0[:, 1], color='blue', marker='s', label="Class 1", alpha=0.8)
    plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='o', label="Class -1", alpha=0.8)

    # Vẽ đường phân chia SVM
    x_vals = np.linspace(0, 5, 100)
    y_vals = -w_sklearn[0] / w_sklearn[1] * x_vals - b_sklearn / w_sklearn[1]
    margin1 = y_vals - 1 / w_sklearn[1]
    margin2 = y_vals + 1 / w_sklearn[1]

    plt.plot(x_vals, y_vals, 'k-', linewidth=2, label="Decision Boundary")
    plt.plot(x_vals, margin1, 'k--', linewidth=1)
    plt.plot(x_vals, margin2, 'k--', linewidth=1)

    # Tô màu vùng phân loại
    plt.fill_between(x_vals, y_vals, margin2, color='blue', alpha=0.1)
    plt.fill_between(x_vals, margin1, color='red', alpha=0.1)

    plt.xlim(1, 5)
    plt.ylim(0, 5)
    plt.xticks([])
    plt.yticks([])

    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.title('SKLearn SVM', fontsize=15)
    plt.legend()

    pdf.savefig()
    plt.show()
