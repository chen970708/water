# -*- coding: utf-8 -*-
# 導入必要的庫
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 假設的學習時間與分數數據
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # 學習時間（小時）
scores = np.array([5, 12, 26, 35, 44, 52, 60, 62, 70, 73])           # 考試分數

# 分割數據為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(study_hours, scores, test_size=0.2, random_state=42)

# 創建隨機森林回歸模型並進行訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 打印模型結果
print("模型的預測結果：")
print("訓練集分數預測:", model.predict(X_train))
print("測試集分數預測:", model.predict(X_test))

# 使用者輸入需要預測的學習時間
print("請輸入想要預測的學習時間（小時），用逗號分隔，例如：1,5,10:")
input_hours = input()
predicted_hours = np.array([float(x) for x in input_hours.split(",")]).reshape(-1, 1)

# 預測輸入數據的分數
predicted_scores = model.predict(predicted_hours)

# 打印預測結果
print("預測結果:")
for i in range(len(predicted_hours)):
    print(f"學習時間: {predicted_hours[i][0]} 小時, 預測分數: {predicted_scores[i]:.2f}")

# 可視化結果
plt.scatter(study_hours, scores, color="blue", label="Actual Data")
plt.plot(study_hours, model.predict(study_hours), color="red", label="Model Prediction")
plt.scatter(predicted_hours, predicted_scores, color="green", label="Predictions (Input)")
plt.xlabel("Study Hours (Hours)")
plt.ylabel("Exam Scores")
plt.legend()
plt.title("Random Forest Regression: Study Hours vs Exam Scores")
plt.show()
