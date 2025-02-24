import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

file_path = "k_Nearest_Neighbors.csv"
dataFile = pd.read_csv(file_path)

selected_columns = ["Diện tích (m²)", "Giá nhà (triệu đồng)"]
data = dataFile[selected_columns].dropna().values

data_t = data.T

# Áp dụng thuật toán Fuzzy C-Means
n_clusters = 3
cntr, u, _, _, _, _, _ = fuzz.cmeans(
    data_t, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
labels = np.argmax(u, axis=0)
dataFile["Cluster"] = labels

plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')

# Vẽ tâm cụm
plt.scatter(cntr[:, 0], cntr[:, 1], s=200, marker="X", c='red', label="Cluster Centers")

plt.legend()
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.title("Fuzzy C-Means Clustering")
plt.show()