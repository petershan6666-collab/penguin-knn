import os
os.system('cls')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 讀取 CSV
df = pd.read_csv(
    r"C:\Users\Dan\Desktop\Master's classes\資料探勘\penguins_lter.csv"
)

# 選擇特徵
features = [
    'Culmen Length (mm)',
    'Culmen Depth (mm)',
    'Flipper Length (mm)',
    'Body Mass (g)'
]

X = df[features]

# 預測目標
y = df['Species']

# 顯示資料筆數
print("資料筆數：", len(X))

# 處理缺失值
X = X.dropna()
y = y[X.index]

# 顯示剩餘資料筆數
print("剩餘資料筆數：", len(X))

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切訓練 / 測試資料
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# 比較不同 K 值
for k in [3, 5, 7]:

    # 建立 KNN
    knn = KNeighborsClassifier(n_neighbors=k)

    # 訓練模型
    knn.fit(X_train, y_train)

    # 預測
    y_pred = knn.predict(X_test)

    # 計算正確率
    acc = accuracy_score(y_test, y_pred)

    print(f"K={k} 的正確率：{acc}")