# 🩺🏠 AI Portal — Dự Đoán Tiểu Đường & Dự Đoán Giá Nhà

Hệ thống gồm 2 ứng dụng web độc lập, xây dựng bằng **FastAPI**:

| Ứng dụng | File chạy | Mô hình | Cổng (port) |
|---|---|---|---|
| Dự đoán tiểu đường | `app_diabetes.py` | Random Forest (pipeline) | `8001` |
| Dự đoán giá nhà | `app_housing.py` | Gradient Boosting Regressor | `8002` |

Mỗi ứng dụng có notebook huấn luyện tương ứng để tạo ra model `.pkl` trước khi chạy app.

---

## 1. Cấu trúc thư mục

Sắp xếp các file đã có theo cấu trúc sau (tạo các thư mục còn thiếu):

```
project/
├── app_diabetes.py
├── app_housing.py
├── diabetes_classification.ipynb
├── house_price_prediction.ipynb
├── diabetes.csv
├── vietnam_housing_dataset.csv
├── data/                      # sẽ tự được tạo khi chạy app_housing.py lần đầu
├── models/                    # chứa các file .pkl sau khi train
└── requirements.txt
```

---

## 2. Yêu cầu môi trường

- Python **3.9+**
- pip

### Cài đặt thư viện

Tạo file `requirements.txt` với nội dung:

```
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
jupyter
```

Cài đặt:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Chuẩn bị dữ liệu & huấn luyện model

⚠️ Hai notebook hiện đang đọc dữ liệu từ đường dẫn cứng trên Windows (`C:\DATA\...`). Trước khi chạy, hãy **sửa lại đường dẫn** trong notebook thành đường dẫn tương đối, ví dụ:

**Trong `diabetes_classification.ipynb`:**
```python
# Thay dòng:
diabetes_df = pd.read_csv(r"C:\DATA\diabetes.csv")
# Bằng:
diabetes_df = pd.read_csv("diabetes.csv")
```

**Trong `house_price_prediction.ipynb`:**
```python
# Thay dòng:
housing_df = pd.read_csv(r"C:\DATA\vietnam_housing_dataset.csv")
# Bằng:
housing_df = pd.read_csv("vietnam_housing_dataset.csv")
```

### Chạy notebook để train model

```bash
jupyter notebook
```

Mở lần lượt và **Run All** cho từng notebook:

1. `diabetes_classification.ipynb` → sinh ra trong thư mục `models/`:
   - `diabetes_model_pipeline.pkl` (bắt buộc để chạy app)
   - `diabetes_imputer.pkl`, `diabetes_scaler.pkl`, `diabetes_model.pkl`

2. `house_price_prediction.ipynb` → sinh ra trong thư mục `models/`:
   - `housing_model_gradient_boosting.pkl` (bắt buộc)
   - `housing_scaler.pkl` (bắt buộc)
   - `housing_preprocessor.pkl`

> Nếu không chạy notebook, các app sẽ báo lỗi "Không tìm thấy file model" khi khởi động nhưng vẫn có thể chạy (API sẽ trả lỗi khi gọi `/api/predict`).

---

## 4. Chạy ứng dụng

### 4.1 Ứng dụng Dự đoán Tiểu đường

```bash
python app_diabetes.py
```

- Giao diện web: http://localhost:8001
- Tài liệu API (Swagger): http://localhost:8001/docs
- Kiểm tra trạng thái: http://localhost:8001/api/health

**Input (8 đặc trưng):** số lần mang thai, glucose, huyết áp, độ dày da, insulin, BMI, diabetes pedigree function, tuổi.

### 4.2 Ứng dụng Dự đoán Giá nhà

```bash
python app_housing.py
```

- Giao diện web: http://localhost:8002
- Tài liệu API (Swagger): http://localhost:8002/docs

Lần đầu chạy, app sẽ tự đọc `vietnam_housing_dataset.csv` (đặt cùng thư mục gốc) và tạo `data/clean_housing_data.csv`.

> Có thể chạy đồng thời cả 2 app trên 2 cửa sổ terminal khác nhau vì chúng dùng port khác nhau (8001 và 8002).

---

## 5. Sử dụng API bằng lệnh (tùy chọn)

**Dự đoán tiểu đường:**
```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 80,
    "bmi": 28.0,
    "diabetes_pedigree_function": 0.5,
    "age": 30
  }'
```

**Dự đoán giá nhà:**
```bash
curl -X POST http://localhost:8002/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 60,
    "floors": 3,
    "bedrooms": 3,
    "bathrooms": 2,
    "legal_status": "Sổ đỏ/Sổ hồng",
    "furniture_state": "Đầy đủ"
  }'
```

---

## 6. Truy cập từ điện thoại / thiết bị khác (cùng mạng Wi-Fi)

Cả 2 app đã được cấu hình chạy với `host="0.0.0.0"`, nghĩa là mặc định đã lắng nghe trên mọi địa chỉ mạng — bạn **không cần sửa code**, chỉ cần làm theo các bước sau.

### Bước 1: Đảm bảo máy tính và điện thoại cùng một mạng Wi-Fi

Điện thoại và máy tính chạy app phải kết nối **chung một mạng Wi-Fi** (không dùng 4G/mobile data trên điện thoại).

### Bước 2: Tìm địa chỉ IP nội bộ (LAN IP) của máy tính

**Windows:**
```bash
ipconfig
```
Tìm dòng `IPv4 Address` trong phần Wi-Fi (thường có dạng `192.168.x.x`).

**macOS:**
```bash
ipconfig getifaddr en0
```

**Linux:**
```bash
hostname -I
```
hoặc
```bash
ip addr show | grep "inet "
```

Ví dụ kết quả: `192.168.1.25`

### Bước 3: Truy cập từ điện thoại

Mở trình duyệt trên điện thoại và nhập:

```
http://<LAN-IP-của-máy-tính>:8001   → App Dự đoán Tiểu đường
http://<LAN-IP-của-máy-tính>:8002   → App Dự đoán Giá nhà
```

Ví dụ:
```
http://192.168.1.25:8001
http://192.168.1.25:8002
```

> ⚠️ **Không dùng `localhost` hoặc `127.0.0.1`** trên điện thoại — hai địa chỉ này chỉ trỏ về chính thiết bị đang gõ, không phải máy tính đang chạy server.

### Bước 4: Nếu điện thoại không truy cập được

| Nguyên nhân | Cách khắc phục |
|---|---|
| Firewall (tường lửa) chặn cổng 8001/8002 | **Windows:** vào *Windows Defender Firewall → Cho phép ứng dụng* → thêm Python hoặc mở port. **macOS:** *System Settings → Network → Firewall* → cho phép kết nối đến. **Linux:** `sudo ufw allow 8001` và `sudo ufw allow 8002` |
| Wi-Fi router bật "AP/Client Isolation" (cách ly thiết bị) | Vào trang quản trị router, tắt tính năng cách ly client — tính năng này ngăn các thiết bị trong cùng mạng thấy nhau (thường có ở Wi-Fi công cộng/văn phòng) |
| Máy tính dùng mạng dây (Ethernet) còn điện thoại dùng Wi-Fi khác dải mạng | Đảm bảo cả hai cùng nằm trong một mạng LAN, hoặc lấy đúng IP của card mạng đang thực sự kết nối |
| VPN đang bật trên máy tính | Tắt VPN, vì VPN có thể đổi địa chỉ IP hoặc chặn kết nối LAN |
| IP thay đổi sau khi khởi động lại router | Lấy lại IP bằng lệnh ở Bước 2 mỗi khi kết nối lại Wi-Fi |

### Bước 5 (tùy chọn): Chia sẻ qua Internet mà không cần cùng Wi-Fi

Nếu muốn người khác truy cập từ mạng khác (không cùng Wi-Fi), có thể dùng công cụ tạo đường hầm (tunnel) tạm thời như [ngrok](https://ngrok.com/):

```bash
ngrok http 8001
```

Ngrok sẽ cấp một URL công khai (dạng `https://xxxx.ngrok-free.app`) để truy cập app từ bất kỳ đâu có Internet. Lưu ý: chỉ dùng để demo/thử nghiệm, không phù hợp cho production.

---

## 7. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `❌ Không tìm thấy file model` | Chưa chạy notebook train | Chạy notebook tương ứng để tạo file `.pkl` trong `models/` |
| `Port already in use` | Cổng 8001/8002 đang bị chiếm | Đổi port trong `uvicorn.run(..., port=xxxx)` hoặc tắt tiến trình đang dùng port đó |
| Lỗi đọc CSV trong notebook | Đường dẫn `C:\DATA\...` không tồn tại trên máy bạn | Sửa lại đường dẫn tương đối như hướng dẫn ở mục 3 |
| `ModuleNotFoundError` | Thiếu thư viện | Chạy lại `pip install -r requirements.txt` |

---

## 8. Ghi chú

- Kết quả dự đoán chỉ mang tính **tham khảo**, không thay thế chẩn đoán y tế hoặc thẩm định bất động sản chuyên nghiệp.
- Độ chính xác model tiểu đường hiển thị trên giao diện (~86.36%) được lấy từ kết quả huấn luyện trong notebook; có thể thay đổi nếu train lại với dữ liệu/tham số khác.
