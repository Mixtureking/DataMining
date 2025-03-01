import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Định nghĩa đường dẫn thư mục ảnh
data_dir = r"C:\Users\ADMIN\Desktop\SVM Kernel Image"  # Thư mục chứa dữ liệu

# 2. Tiền xử lý dữ liệu ảnh
image_size = (64, 64)  # Kích thước ảnh cố định
X, y = [], []
class_labels = {}

# Duyệt qua các thư mục con trong data_dir
for label, category in enumerate(os.listdir(data_dir)):
    class_labels[label] = category  # Lưu tên lớp
    category_path = os.path.join(data_dir, category)

    # Đọc từng ảnh trong thư mục con
    for img_name in os.listdir(category_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue  # Bỏ qua file có đuôi .jfif hoặc không phải ảnh

        img_path = os.path.join(category_path, img_name)

        try:
            img = Image.open(img_path).convert("L")  # Đọc ảnh grayscale
            img = img.resize(image_size)  # Resize ảnh về kích thước cố định

            X.append(np.array(img).flatten())  # Chuyển ảnh thành vector
            y.append(label)
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")

X = np.array(X)
y = np.array(y)

# 3. Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Huấn luyện mô hình SVM với kernel RBF
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_model.fit(X_train, y_train)

# 6. Dự đoán và đánh giá
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_labels.values()))

# 7. Hiển thị một số ảnh dự đoán
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img = X_test[i].reshape(image_size)
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {class_labels[y_pred[i]]}")
    ax.axis("off")

plt.show()
