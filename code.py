import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# 1. ĐỌC DỮ LIỆU
df = pd.read_csv('dataset/sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Data: {df['Date'].min()} → {df['Date'].max()} ({len(df)} rows)")

# 2. FEATURE ENGINEERING 

def add_safe_features(df):
    """Chỉ dùng features không cần dữ liệu tương lai"""
    d = df.copy()
    
    # Time features (luôn có sẵn cho tương lai)
    d['year'] = d['Date'].dt.year
    d['month'] = d['Date'].dt.month
    d['day'] = d['Date'].dt.day
    d['dow'] = d['Date'].dt.dayofweek
    d['quarter'] = d['Date'].dt.quarter
    d['day_of_year'] = d['Date'].dt.dayofyear
    d['week_of_year'] = d['Date'].dt.isocalendar().week
    d['days_in_month'] = d['Date'].dt.days_in_month
    d['day_from_end'] = d['days_in_month'] - d['day']
    d['is_month_end'] = d['Date'].dt.is_month_end.astype(int)
    d['is_month_start'] = d['Date'].dt.is_month_start.astype(int)
    d['is_weekend'] = (d['dow'] >= 5).astype(int)
    d['is_quarter_start'] = d['Date'].dt.is_quarter_start.astype(int)
    d['is_quarter_end'] = d['Date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin'] = np.sin(2 * np.pi * d['dow'] / 7)
    d['dow_cos'] = np.cos(2 * np.pi * d['dow'] / 7)
    
    # Time index
    d['time_idx'] = range(len(d))
    
    return d

def add_fourier_features(df):
    d = df.copy()
    dayofyear = d['day_of_year'].values
    for k in range(1, 6):
        d[f'sin_{k}'] = np.sin(2 * np.pi * k * dayofyear / 365.25)
        d[f'cos_{k}'] = np.cos(2 * np.pi * k * dayofyear / 365.25)
    return d

def add_holiday_features(df):
    d = df.copy()
    d['mm_dd'] = d['Date'].dt.strftime('%m-%d')
    
    # Solar holidays
    solar_holidays = ['01-01', '04-30', '05-01', '09-02', '12-31']
    d['is_solar_holiday'] = d['mm_dd'].isin(solar_holidays).astype(int)
    
    pre_solar = ['04-29', '09-01']
    d['is_pre_solar_holiday'] = d['mm_dd'].isin(pre_solar).astype(int)
    
    # Lunar holidays (Tet)
    tet_dates = [
        '2013-02-10', '2013-02-11', '2013-02-12',
        '2014-01-31', '2014-02-01', '2014-02-02',
        '2015-02-19', '2015-02-20', '2015-02-21',
        '2016-02-08', '2016-02-09', '2016-02-10',
        '2017-01-28', '2017-01-29', '2017-01-30',
        '2018-02-16', '2018-02-17', '2018-02-18',
        '2019-02-05', '2019-02-06', '2019-02-07',
        '2020-01-25', '2020-01-26', '2020-01-27',
        '2021-02-12', '2021-02-13', '2021-02-14',
        '2022-02-01', '2022-02-02', '2022-02-03',
        '2023-01-22', '2023-01-23', '2023-01-24',
        '2024-02-10', '2024-02-11', '2024-02-12',
    ]
    tet_dates = [pd.Timestamp(d) for d in tet_dates]
    d['is_tet'] = d['Date'].isin(tet_dates).astype(int)
    
    # Pre-Tet
    pre_tet_dates = [
        '2013-02-08', '2013-02-09',
        '2014-01-29', '2014-01-30',
        '2015-02-17', '2015-02-18',
        '2016-02-06', '2016-02-07',
        '2017-01-26', '2017-01-27',
        '2018-02-14', '2018-02-15',
        '2019-02-02', '2019-02-03', '2019-02-04',
        '2020-01-23', '2020-01-24',
        '2021-02-10', '2021-02-11',
        '2022-01-30', '2022-01-31',
        '2023-01-19', '2023-01-20', '2023-01-21',
        '2024-02-07', '2024-02-08', '2024-02-09',
    ]
    pre_tet_dates = [pd.Timestamp(d) for d in pre_tet_dates]
    d['is_pre_tet'] = d['Date'].isin(pre_tet_dates).astype(int)
    
    # Double days
    d['is_double_day'] = (d['day'] == d['month']).astype(int)
    d['is_double_day'] = np.where(d['month'] < 9, 0, d['is_double_day'])
    
    return d.drop(columns=['mm_dd'])


# 3. ÁP DỤNG FEATURES

print("\nEngineering safe features...")

df = add_safe_features(df)
df = add_fourier_features(df)
df = add_holiday_features(df)

# Features list
FEATURES = [col for col in df.columns if col not in ['Date', 'Revenue', 'COGS', 'mm_dd']]
print(f"📊 Total features (safe): {len(FEATURES)}")


# 4. WALK-FORWARD VALIDATION
print("\nTraining with walk-forward validation...")

# Đảm bảo data được train theo thứ tự thời gian
df = df.sort_values('Date')

# Split: train 2012-2019, val 2020, test 2021
train_df = df[df['year'] <= 2019].copy()
val_df = df[df['year'] == 2020].copy()
test_df = df[df['year'] == 2021].copy()
final_train_df = df[df['year'] <= 2021].copy()

print(f"   Train (2012-2019): {len(train_df)} rows")
print(f"   Validation (2020): {len(val_df)} rows")
print(f"   Test (2021): {len(test_df)} rows")
print(f"   Final Train (2012-2021): {len(final_train_df)} rows")

# Scale revenue
scaler = StandardScaler()
train_df['Revenue_scaled'] = scaler.fit_transform(train_df[['Revenue']])
val_df['Revenue_scaled'] = scaler.transform(val_df[['Revenue']])
test_df['Revenue_scaled'] = scaler.transform(test_df[['Revenue']])
final_train_df['Revenue_scaled'] = scaler.transform(final_train_df[['Revenue']])

X_train = train_df[FEATURES]
y_train = train_df['Revenue_scaled']
X_val = val_df[FEATURES]
y_val = val_df['Revenue_scaled']
X_test = test_df[FEATURES]
y_test = test_df['Revenue_scaled']
X_final_train = final_train_df[FEATURES]
y_final_train = final_train_df['Revenue_scaled']

# 5. TRAIN MODEL WITH WALK-FORWARD
print("\nTraining LightGBM with walk-forward validation...")

model = lgb.LGBMRegressor(
    objective='mae',
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# Train on 2012-2019, validate on 2020
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    eval_metric='mae', 
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Test on 2021
pred_test_scaled = model.predict(X_test)
pred_test = scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
mae_2021 = mean_absolute_error(test_df['Revenue'].values, pred_test)
print(f"\n📊 WALK-FORWARD VALIDATION 2021 MAE: {mae_2021:>10,.0f}")

# 6. FINAL MODEL ON ALL TRAINING DATA
print("\n🚀 Training final model on 2012-2021...")

final_model = lgb.LGBMRegressor(
    objective='mae',
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

final_model.fit(X_final_train, y_final_train)

# Validate on test set 2021
final_pred_scaled = final_model.predict(X_test)
final_pred = scaler.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
final_mae_2021 = mean_absolute_error(test_df['Revenue'].values, final_pred)
print(f"   Final model MAE on 2021: {final_mae_2021:>10,.0f}")

# 7. DỰ BÁO 2023-2024
print("\n🔮 Generating forecasts for 2023-2024...")

sample = pd.read_csv('dataset/sample_submission.csv')
sample['Date'] = pd.to_datetime(sample['Date'])

# Tạo features cho sample
sample = add_safe_features(sample)
sample = add_fourier_features(sample)
sample = add_holiday_features(sample)

X_future = sample[FEATURES]

# Dự báo scaled
pred_future_scaled = final_model.predict(X_future)

# Inverse transform về revenue thực
pred_future = scaler.inverse_transform(pred_future_scaled.reshape(-1, 1)).flatten()

# SỬA LỖI: dùng np.clip thay vì .clip()
sample['Revenue'] = np.clip(pred_future, 0, None)

# COGS dựa trên tỷ lệ trung bình 3 năm gần nhất
cogs_ratio_2019 = df[df['year'] == 2019]['COGS'].sum() / df[df['year'] == 2019]['Revenue'].sum()
cogs_ratio_2020 = df[df['year'] == 2020]['COGS'].sum() / df[df['year'] == 2020]['Revenue'].sum()
cogs_ratio_2021 = df[df['year'] == 2021]['COGS'].sum() / df[df['year'] == 2021]['Revenue'].sum()
avg_cogs_ratio = (cogs_ratio_2019 + cogs_ratio_2020 + cogs_ratio_2021) / 3

sample['COGS'] = sample['Revenue'] * avg_cogs_ratio
sample['Revenue'] = sample['Revenue'].round(2)
sample['COGS'] = sample['COGS'].round(2)

submission = sample[['Date', 'Revenue', 'COGS']]
submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
submission.to_csv('submission.csv', index=False)

print(f"\n✅ Đã lưu submission.csv ({len(submission)} dòng)")

# Thống kê
forecast_2023 = submission[submission['Date'].str[:4] == '2023']['Revenue'].sum()
forecast_2024 = submission[submission['Date'].str[:4] == '2024']['Revenue'].sum()
forecast_2024_h1 = submission[(submission['Date'].str[:4] == '2024') & 
                               (submission['Date'].str[5:7].astype(int) <= 6)]['Revenue'].sum()

print(f"\n📊 Forecast totals:")
print(f"   2023 Revenue:        {forecast_2023:>18,.0f}")
print(f"   2024 H1 Revenue:     {forecast_2024_h1:>18,.0f}")
print(f"   2024 Full Revenue:   {forecast_2024:>18,.0f}")

# Feature importance
print("\n📊 Top 15 Feature Importance:")
importance_df = pd.DataFrame({
    'feature': FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

for _, row in importance_df.iterrows():
    print(f"   {row['feature']:<25}: {row['importance']:.0f}")


