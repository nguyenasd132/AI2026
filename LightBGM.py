import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Thiết lập phong cách đồ họa chuyên nghiệp cho Analyst
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

print("Đang tải dữ liệu gốc...")
df = pd.read_csv("Dataset.csv", encoding='utf-8-sig')
embeddings = np.load("Bertdataset.npy") 

print(f"[+] Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")

# Lọc Giá (1 -> 60 tỷ) và Diện tích (0 -> 1000m2)
mask_price_area = (df['Price_Billion'] <= 60) & (df['Price_Billion'] >= 1) & (df['Area'] > 0) & (df['Area'] <= 1000)

# Bộ lọc từ khóa: Loại trừ các giao dịch sai phân khúc
keywords = r'cho thuê|bán đất|đất nền|đất trống|nhà trọ|phòng trọ'
mask_text = ~df['Title'].str.contains(keywords, case=False, na=False)

# Áp dụng bộ lọc kép
final_mask = mask_price_area & mask_text
df = df[final_mask].copy()
embeddings = embeddings[final_mask]

print(f"[+] Kích thước dữ liệu sau làm sạch: {df.shape[0]} dòng (Giữ lại {(df.shape[0]/8461)*100:.1f}%)")

print("\n--- THỐNG KÊ MÔ TẢ ĐA CHIỀU ---")
display(df[['Price_Billion', 'Area', 'Bedrooms', 'Bathrooms']].describe().round(2))

plt.figure(figsize=(9, 5))

# Biểu đồ phân bố (Histogram)
sns.histplot(df['Price_Billion'], bins=50, kde=True, color='royalblue')
plt.title('Phân bố Giá Nhà', fontsize=14)
plt.xlabel('Giá (Tỷ VND)')
plt.ylabel('Số lượng')
plt.axvline(df['Price_Billion'].mean(), color='red', linestyle='--', label=f"Trung bình: {df['Price_Billion'].mean():.1f} Tỷ")
plt.axvline(df['Price_Billion'].median(), color='green', linestyle='-', label=f"Trung vị: {df['Price_Billion'].median():.1f} Tỷ")
plt.legend()

plt.show()

plt.figure(figsize=(9, 5))
# Chuyển về biểu đồ Scatter cơ bản, dễ nhìn
sns.scatterplot(data=df, x='Area', y='Price_Billion', color='crimson', alpha=0.6)
plt.title('Mối quan hệ: Diện Tích và Giá Nhà', fontsize=14)
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá (Tỷ VND)')
plt.show()

plt.figure(figsize=(7, 5))
corr_matrix = df[['Price_Billion', 'Area', 'Bedrooms', 'Bathrooms']].corr()

# Vẽ Heatmap nguyên bản, đơn giản
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
plt.title('Bản đồ Tương Quan (Heatmap)', fontsize=14)
plt.show()

# Lấy ra các đặc trưng số
numeric_cols = ['Area', 'Bedrooms', 'Bathrooms']
numeric_features = df[numeric_cols].values

# Chuẩn hóa (Scale) các đặc trưng số để thuật toán không bị thiên lệch bởi đơn vị đo
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

# Nối ma trận dữ liệu số đã chuẩn hóa với siêu ma trận BERT embeddings
X = np.hstack((numeric_scaled, embeddings))
y = df['Price_Billion'].values

print(f"[+] Kích thước ma trận Features (X): {X.shape} (Gồm 3 biến số + 768 biến BERT)")
print(f"[+] Kích thước Vector Mục tiêu (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo định dạng lgb.Dataset giúp LightGBM quản lý bộ nhớ và tính toán Histogram siêu tốc
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 45,
    'max_depth': 10,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbose': -1
}

print("\n🚀 Bắt đầu huấn luyện LightGBM...")
start_time = time.time()

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1500,                
    valid_sets=[train_data, test_data],  
    callbacks=[
        lgb.early_stopping(stopping_rounds=50), 
        lgb.log_evaluation(period=100)          
    ]
)

end_time = time.time()
print(f"\n⏳ Hoàn thành trong: {end_time - start_time:.2f} giây")

# =========================================================
# ĐÁNH GIÁ CHỈ SỐ KPI
# =========================================================
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
mae = mean_absolute_error(y_test, y_pred)          
r2 = r2_score(y_test, y_pred)                      

print(f"\n🎯 --- KẾT QUẢ ĐÁNH GIÁ MODEL ---")
print(f"✅ R-Squared (R2): {r2:.4f} (Mô hình giải thích được {r2*100:.2f}% sự biến động của giá nhà)")
print(f"✅ RMSE: {rmse:.4f} Tỷ (Độ lệch chuẩn tổng thể)")
print(f"✅ MAE:  {mae:.4f} Tỷ (Lệch tuyệt đối trung bình trên mỗi căn)")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='darkorange', edgecolor='w', s=60)

# Đường chéo 45 độ (Đường hoàn hảo)
max_val = max(max(y_test), max(y_pred))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2)

plt.title('Độ hội tụ: Giá Thực Tế vs Giá Dự Đoán', fontweight='bold')
plt.xlabel('Giá Thực Tế (Tỷ VND)')
plt.ylabel('Giá Dự Đoán bởi LightGBM (Tỷ VND)')
plt.show()