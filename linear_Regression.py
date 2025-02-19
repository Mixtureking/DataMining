from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
# Dữ liệu mẫu
X = np.linspace(30, 80, 50).reshape(-1,1)

noise = np.random.randint(-10,10, size=(50,))
y = 5 * X.flatten() + 100 + noise

# Tạo mô hình và huấn luyện
model = LinearRegression()
model.fit(X, y)

# Lấy hệ số hồi quy và hệ số chặn
m = model.coef_[0]   # Hệ số hồi quy (slope)
b = model.intercept_ # Hệ số chặn (y-intercept)

predicted_price = model.predict([[65]])

# Lấy Data nhỏ (5 cặp X,y)
x_mean = np.mean(X[:5])
y_mean = np.mean(y[:5])
m = np.sum((X[:5] - x_mean) * (y[:5] - y_mean)) / np.sum((X[:5] - x_mean) ** 2)
b = y_mean - m * x_mean
df5 = pd.DataFrame({"Diện tích (m²)": X[:5].flatten(), "Giá nhà (triệu đồng)": y[:5]})
print(df5.to_string())
print(f"Hệ số hồi quy (m) của 5 số đầu: {m}")
print(f"Hệ số chặn (b) của 5 số sau: {b}")

# Lấy Data lớn (50 cặp X,y)
df = pd.DataFrame({"Diện tích (m²)": X.flatten(), "Giá nhà (triệu đồng)": y})

print(df.to_string())

print(f"Hệ số hồi quy: {model.coef_[0]}, Hệ số chặn: {model.intercept_}")
print(f"Dự đoán giá nhà cho 65m2: {predicted_price[0]:.2f} triệu đồng")

df.to_csv("linear_Regression.csv", index=False, encoding="utf-8-sig")
