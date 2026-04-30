# Demand Forecasting Strategy – Retail (Vietnam Fashion E-commerce)

## 1. Mục tiêu

Dự báo doanh thu (Revenue) theo ngày trong giai đoạn tương lai nhằm:

* tối ưu phân bổ tồn kho
* lập kế hoạch khuyến mãi
* hỗ trợ logistics

---

## 2. Khung tư duy (Business-driven)

Doanh thu được mô hình hóa theo 3 thành phần chính:

Revenue = Seasonality + Trend + External Drivers

### 2.1 Seasonality (Tính mùa vụ)

* Hành vi mua sắm lặp lại theo chu kỳ:

  * theo tuần (weekday vs weekend)
  * theo năm (mùa cao điểm, cuối năm)
* Đây là yếu tố ổn định và dễ dự báo nhất

---

### 2.2 Trend (Xu hướng dài hạn)

* Tăng trưởng hoặc suy giảm theo thời gian
* Có thể bị ảnh hưởng bởi các sự kiện lớn (ví dụ: COVID)

---

### 2.3 External Drivers (Yếu tố tác động)

Các yếu tố làm doanh thu lệch khỏi chu kỳ:

* Promotions:

  * số lượng chương trình khuyến mãi
  * mức giảm giá
* Web traffic:

  * sessions, unique visitors
  * phản ánh nhu cầu trước khi mua
* Orders:

  * số đơn hàng
  * tổng số lượng sản phẩm bán ra

---

## 3. Feature Engineering

### 3.1 Seasonality features

* day_of_week
* month / day_of_year
* cyclic encoding (sin/cos)

---

### 3.2 Lag features

* lag_7, lag_14 (ngắn hạn)
* lag_30 (trung hạn)
* lag_365 (chu kỳ năm)

---

### 3.3 Rolling features

* rolling_mean_7, rolling_mean_30
* rolling_std

---

### 3.4 External features (aggregate theo ngày)

* số promotion active
* traffic (sessions)
* số đơn hàng

---

## 4. Mô hình

Sử dụng LightGBM do:

* phù hợp với dữ liệu dạng bảng
* học tốt nonlinear pattern
* hiệu quả cao với feature engineering

---

## 5. Phương pháp dự báo

* Sử dụng recursive forecasting:

  * dự báo từng ngày
  * sử dụng kết quả trước đó làm input cho ngày tiếp theo

---

## 6. Validation

* Chia dữ liệu theo thời gian (time-based split)
* Tránh dùng random split để không gây leakage

---

## 7. Insight quan trọng

* Doanh thu có tính mùa vụ rõ rệt và ổn định
* Các biến động lớn (ví dụ COVID) là yếu tố ngoại sinh
* External drivers (traffic, promotion) giúp giải thích spike

---

## 8. Kết luận

Mô hình dự báo hiệu quả cần:

* khai thác seasonality (pattern lặp lại)
* kết hợp trend (xu hướng dài hạn)
* bổ sung external drivers (yếu tố tác động)

=> Không chỉ dựa vào dữ liệu quá khứ mà còn phản ánh đúng hành vi kinh doanh thực tế
