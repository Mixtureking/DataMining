# Import các thư viện cần thiết
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Tải và chuẩn bị dữ liệu
# 1. Đọc dữ liệu từ file CSV
metVuong = np.linspace(6000, 10000, 500).reshape(-1, 1)
noise = np.random.randint(-750,750, size=(500,))
giaNha = 5 * metVuong.flatten() + 100 + noise
# Đọc file CSV
df = pd.DataFrame({"Diện tích (m²)": metVuong.flatten(), "Giá nhà (triệu đồng)": giaNha})
df.to_csv("k_Nearest_Neighbors.csv", index=False, encoding="utf-8-sig")
# 2. Lấy các cột cần phân loại
X = df[['Diện tích (m²)', 'Giá nhà (triệu đồng)']].values  # Đặc trưng

df['Nhãn'] = df['Giá nhà (triệu đồng)'].apply(lambda x: 'Giá rẻ' if x < 35000 else ('Giá trung bình' if x < 40000 else 'Giá đắt'))
y = df['Nhãn'].values
# 3. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Tìm giá trị K tối ưu bằng Cross-Validation
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Xác định giá trị K với độ chính xác cao nhất
optimal_k = k_range[np.argmax(k_scores)]
print(f'Giá trị K tối ưu: {optimal_k}')

# 6. Vẽ biểu đồ để quan sát mối quan hệ giữa K và độ chính xác
plt.plot(k_range, k_scores)
plt.xlabel('Giá trị của K')
plt.ylabel('Độ chính xác trung bình')
plt.title('Độ chính xác trung bình theo từng giá trị của K')
plt.show()

# 7. Huấn luyện mô hình KNN với K tối ưu
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# 8. Đánh giá mô hình trên tập kiểm tra
accuracy = knn.score(X_test, y_test)
print(f'Độ chính xác của mô hình trên tập kiểm tra: {accuracy * 100:.2f}%')

# 9. Hiển thị ma trận nhầm lẫn
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=['Giá rẻ', 'Giá trung bình', 'Giá đắt'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Giá rẻ', 'Giá trung bình', 'Giá đắt'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Ma trận nhầm lẫn')
plt.show()


# 10. Trực quan hóa vùng quyết định

def plot_decision_boundary(X, y, model, title):
    h = 1  # Bước của lưới lớn hơn để tránh lỗi bộ nhớ
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = scaler.transform(grid_points)  # Chuẩn hóa điểm lưới

    # Dự đoán nhãn
    Z_labels = model.predict(grid_points)

    # Chuyển đổi nhãn thành số
    label_mapping = {'Giá rẻ': 0, 'Giá trung bình': 1, 'Giá đắt': 2}
    Z = np.array([label_mapping[label] for label in Z_labels])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0, cmap=plt.cm.Paired)
    scatter_colors = [label_mapping[label] for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=scatter_colors, edgecolor='k', cmap=plt.cm.Paired)

    plt.xlabel('Diện tích (m²)')
    plt.ylabel('Giá nhà (triệu đồng)')
    plt.title(title)
    plt.show()


# Vẽ lại vùng quyết định
plot_decision_boundary(X_test, y_test, knn, 'Biểu đồ vùng quyết định của KNN')