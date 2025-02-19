import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
df = pd.read_csv("linear_Regression.csv")  # Đọc file CSV
X = df[['Diện tích (m²)','Giá nhà (triệu đồng)']].values  # Lấy các cột cần phân cụm



# # 1️⃣ Tìm số cụm tối ưu bằng phương pháp Elbow
# inertia = []
# K_range = range(1, 11)  # Kiểm tra từ K=1 đến K=10
#
# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)
#
# # Vẽ biểu đồ Elbow
# plt.figure(figsize=(6, 4))
# plt.plot(K_range, inertia, marker='o', linestyle='-')
# plt.xlabel('Số cụm (K)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method - Tìm số cụm tối ưu')
# plt.show()

# 2️⃣ Chọn số cụm bằng phương pháp Silhouette Score
silhouette_scores = []
for k in range(2, 11):  # Bắt đầu từ 2 cụm trở lên
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Tìm K có silhouette score cao nhất
optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"Số cụm tối ưu theo Silhouette Score: {optimal_k}")

# 3️⃣ Áp dụng K-Means với số cụm tối ưu
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_final = kmeans_final.fit_predict(X)

# Vẽ biểu đồ phân cụm
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=labels_final, cmap='viridis', edgecolor='k')
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.legend()
plt.title(f"K-means Clustering (K={optimal_k})")
plt.show()
