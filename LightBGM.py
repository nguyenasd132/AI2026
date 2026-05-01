import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Đọc dữ liệu của bạn (ví dụ từ file CSV hoặc Excel)
# Thay đường dẫn 'data.csv' bằng file dữ liệu thực tế của bạn
# df = pd.read_csv('data.csv')

# Xác định cột mục tiêu (target) - cột chứa giá nhà cần dự đoán
# target_col = 'GiaNha' 

# X = df.drop(columns=[target_col]) # Tập đặc trưng (features)
# y = df[target_col]                # Tập nhãn (target)

# --- Dữ liệu giả lập để script có thể chạy thử (bạn hãy xóa phần này khi dùng dữ liệu thật) ---
df = pd.DataFrame({
    'DienTich': np.random.randint(30, 200, 1000),
    'SoPhongNgu': np.random.randint(1, 5, 1000),
    'TuoiNha': np.random.randint(1, 50, 1000),
    'GiaNha': np.random.rand(1000) * 5000 + 1000
})
target_col = 'GiaNha'
X = df.drop(columns=[target_col])
y = df[target_col]
# ----------------------------------------------------------------------------------------------

# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tạo Dataset cho LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. Thiết lập các siêu tham số (Hyperparameters) cho mô hình
params = {
    'objective': 'regression',       # Bài toán hồi quy (dự đoán giá)
    'metric': 'rmse',                # Root Mean Squared Error
    'boosting_type': 'gbdt',         # Gradient Boosting Decision Tree
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'verbose': -1
}

# 5. Huấn luyện mô hình
print("Bắt đầu huấn luyện mô hình LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 6. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 7. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Kết quả đánh giá ---")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# (Tùy chọn) In độ quan trọng của các đặc trưng
lgb.plot_importance(model, max_num_features=10, importance_type='split')
# plt.show() # Uncomment nếu bạn có import matplotlib.pyplot as plt
