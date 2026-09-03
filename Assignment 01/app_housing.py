# -*- coding: utf-8 -*-
"""
Ứng dụng Dự đoán Giá nhà
AI Portal - House Price Prediction System
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. KHỞI TẠO ỨNG DỤNG
# ============================================================

app = FastAPI(
    title="AI Portal - Dự Đoán Giá Nhà",
    description="Ứng dụng thẩm định giá bất động sản dựa trên các đặc điểm của căn nhà",
    version="1.0.0"
)

# ============================================================
# 2. TẢI MODEL VÀ DỮ LIỆU
# ============================================================

housing_model = None
housing_scaler = None

try:
    housing_scaler = joblib.load("models/housing_scaler.pkl")
    housing_model = joblib.load("models/housing_model_gradient_boosting.pkl")
    print("✅ Đã tải model dự đoán giá nhà thành công!")
except FileNotFoundError:
    print("❌ Không tìm thấy file model. Vui lòng chạy notebook để train model trước.")
    print("   Đường dẫn cần có: models/housing_scaler.pkl")
    print("   Đường dẫn cần có: models/housing_model_gradient_boosting.pkl")
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")

# --- Dữ liệu nhà đất ---
db_df = pd.DataFrame()

try:
    db_df = pd.read_csv("data/clean_housing_data.csv")
    db_df["id"] = db_df.index
    print(f"✅ Đã tải dữ liệu nhà đất: {len(db_df)} căn")
except FileNotFoundError:
    try:
        df_raw = pd.read_csv("vietnam_housing_dataset.csv")
        db_df = df_raw[['Address', 'Area', 'Floors', 'Bedrooms', 'Bathrooms',
                        'House direction', 'Legal status', 'Furniture state', 'Price']].copy()
        db_df = db_df.dropna(subset=['Price'])
        db_df = db_df.reset_index(drop=True)
        db_df["id"] = db_df.index
        db_df["Clean_Address"] = db_df["Address"].str.lower()
        os.makedirs("data", exist_ok=True)
        db_df.to_csv("data/clean_housing_data.csv", index=False)
        print(f"✅ Đã tạo và tải dữ liệu nhà đất: {len(db_df)} căn")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")


# ============================================================
# 3. ĐỊNH NGHĨA REQUEST MODELS
# ============================================================

class SearchFilter(BaseModel):
    address: str = ""
    min_area: float = 30
    max_area: float = 120
    bedrooms: int = 0
    bathrooms: int = 0
    floors: int = 0
    house_direction: str = "Tất cả"
    legal_status: str = "Tất cả"
    max_budget: float = 20.0


class ValuationRequest(BaseModel):
    house_id: int


class PredictRequest(BaseModel):
    area: float
    floors: float
    bedrooms: float
    bathrooms: float
    legal_status: str
    furniture_state: str


# ============================================================
# 4. HTML CONTENT (giữ nguyên)
# ============================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Dự Đoán Giá Nhà - AI Portal</title>
    <style>
        :root { --primary: #2563eb; --hover: #1d4ed8; --bg: #f0f4f8; --card: #ffffff; --text: #0f172a; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 20px; }

        .container { max-width: 1350px; width: 100%; }

        .header { text-align: center; margin-bottom: 24px; }
        .header h1 { font-size: 1.8rem; color: var(--primary); font-weight: 700; }
        .header p { font-size: 0.9rem; color: #64748b; margin-top: 4px; }
        .header .badge { display: inline-block; background: #dbeafe; color: #1d4ed8; padding: 2px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; margin-top: 8px; }

        .main-layout { display: grid; grid-template-columns: 340px 1fr; gap: 20px; }
        @media(max-width: 900px) { .main-layout { grid-template-columns: 1fr; } }

        .filter-panel { background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; height: fit-content; }
        .filter-panel h2 { font-size: 1rem; margin-bottom: 16px; color: #1e293b; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px; }

        .form-group { margin-bottom: 12px; }
        label { display: block; font-size: 0.78rem; font-weight: 600; color: #475569; margin-bottom: 4px; }
        input, select { width: 100%; padding: 8px 10px; border: 1.5px solid #e2e8f0; border-radius: 6px; font-size: 0.85rem; outline: none; background: #f8fafc; }
        input:focus, select:focus { border-color: var(--primary); background: #fff; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

        .btn-search { width: 100%; padding: 12px; background: var(--primary); color: #fff; border: none; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; margin-top: 8px; transition: background 0.2s; }
        .btn-search:hover { background: var(--hover); }
        .btn-search:disabled { opacity: 0.6; cursor: not-allowed; }

        .results-panel { display: flex; flex-direction: column; gap: 14px; }
        .results-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; flex-wrap: wrap; }
        .results-header h3 { font-size: 1rem; }
        .results-header span { font-size: 0.8rem; color: #64748b; }

        .house-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }

        .house-card { background: var(--card); border-radius: 10px; border: 1px solid #e2e8f0; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 2px 4px rgba(0,0,0,0.03); }
        .card-header { padding: 12px 14px; border-bottom: 1px solid #f8fafc; background: #fafafa; }
        .listed-price { font-size: 1.25rem; font-weight: 700; color: #dc2626; }
        .address { font-size: 0.78rem; color: #64748b; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        .card-body { padding: 12px 14px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
        .specs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
        .spec-chip { background: #f1f5f9; padding: 3px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; color: #334155; }

        .details { font-size: 0.75rem; color: #475569; line-height: 1.5; margin-bottom: 10px; }

        .btn-evaluate { width: 100%; padding: 8px; background: var(--primary); color: #fff; border: none; border-radius: 6px; font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        .btn-evaluate:hover { background: var(--hover); }
        .btn-evaluate:disabled { opacity: 0.6; cursor: not-allowed; }

        .ai-result-box { margin-top: 8px; padding: 10px 12px; border-radius: 6px; display: none; }
        .ai-result-box.cheap { display: block; background: #ecfdf5; border: 1px solid #86efac; }
        .ai-result-box.fair { display: block; background: #eff6ff; border: 1px solid #93c5fd; }
        .ai-result-box.expensive { display: block; background: #fef2f2; border: 1px solid #fca5a5; }

        .ai-price { font-size: 1rem; font-weight: 700; }
        .ai-tag { font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-top: 4px; }
        .cheap .ai-tag { background: #dcfce7; color: #15803d; }
        .fair .ai-tag { background: #dbeafe; color: #1d4ed8; }
        .expensive .ai-tag { background: #fee2e2; color: #b91c1c; }
        .ai-desc { font-size: 0.72rem; margin-top: 4px; color: #334155; }

        .empty-state { text-align: center; padding: 40px 20px; color: #64748b; background: #fff; border-radius: 8px; border: 1px dashed #cbd5e1; grid-column: 1/-1; font-size: 0.9rem; }

        .footer { text-align: center; margin-top: 30px; font-size: 0.7rem; color: #94a3b8; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🏠 Dự Đoán Giá Nhà</h1>
        <p>Tìm kiếm và thẩm định giá bất động sản bằng trí tuệ nhân tạo</p>
        <span class="badge">🤖 Mô hình Gradient Boosting • R² = 0.2826</span>
    </div>

    <div class="main-layout">
        <div class="filter-panel">
            <h2>🔍 Bộ Lọc Tìm Kiếm</h2>
            <form id="filterForm">
                <div class="form-group">
                    <label>Địa chỉ / Khu vực</label>
                    <input type="text" id="address" placeholder="VD: Hà Nội, Đống Đa..." value="Hà Nội">
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>DT Tối thiểu (m²)</label>
                        <input type="number" id="min_area" value="30" required>
                    </div>
                    <div class="form-group">
                        <label>DT Tối đa (m²)</label>
                        <input type="number" id="max_area" value="120" required>
                    </div>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Phòng ngủ</label>
                        <input type="number" id="bedrooms" value="2" required>
                    </div>
                    <div class="form-group">
                        <label>Phòng tắm</label>
                        <input type="number" id="bathrooms" value="2" required>
                    </div>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>Số tầng</label>
                        <input type="number" id="floors" value="3" required>
                    </div>
                    <div class="form-group">
                        <label>Hướng nhà</label>
                        <select id="house_direction">
                            <option value="Tất cả">Tất cả</option>
                            <option value="Đông - Nam">Đông - Nam</option>
                            <option value="Nam">Nam</option>
                            <option value="Đông">Đông</option>
                            <option value="Tây - Nam">Tây - Nam</option>
                            <option value="Bắc">Bắc</option>
                            <option value="Tây">Tây</option>
                            <option value="Đông - Bắc">Đông - Bắc</option>
                            <option value="Tây - Bắc">Tây - Bắc</option>
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Pháp lý</label>
                    <select id="legal_status">
                        <option value="Tất cả">Tất cả</option>
                        <option value="Have certificate">Có sổ</option>
                        <option value="Sale contract">Hợp đồng mua bán</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Ngân sách tối đa (Tỷ VNĐ)</label>
                    <input type="number" step="0.5" id="max_budget" value="10.0" required>
                </div>
                <button type="submit" class="btn-search" id="btnSubmit">🔍 Lọc Bất Động Sản</button>
            </form>
        </div>

        <div class="results-panel">
            <div class="results-header">
                <h3>Danh Sách Nhà Phù Hợp</h3>
                <span id="resultCount">(0 căn)</span>
            </div>
            <div class="house-grid" id="gridContainer"></div>
        </div>
    </div>

    <div class="footer">
        <span>⚠️ Kết quả mang tính tham khảo, không thay thế đánh giá chuyên môn</span>
    </div>
</div>

<script>
    document.getElementById('filterForm').addEventListener('submit', (e) => {
        e.preventDefault();
        triggerSearch();
    });

    async function triggerSearch() {
        const btn = document.getElementById('btnSubmit');
        btn.innerText = "⏳ Đang tìm...";
        btn.disabled = true;

        const payload = {
            address: document.getElementById('address').value,
            min_area: parseFloat(document.getElementById('min_area').value),
            max_area: parseFloat(document.getElementById('max_area').value),
            bedrooms: parseInt(document.getElementById('bedrooms').value),
            bathrooms: parseInt(document.getElementById('bathrooms').value),
            floors: parseInt(document.getElementById('floors').value),
            house_direction: document.getElementById('house_direction').value,
            legal_status: document.getElementById('legal_status').value,
            max_budget: parseFloat(document.getElementById('max_budget').value)
        };

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const houses = await res.json();
            renderHouses(houses);
        } catch(e) {
            alert("Lỗi kết nối máy chủ!");
        } finally {
            btn.innerText = "🔍 Lọc Bất Động Sản";
            btn.disabled = false;
        }
    }

    function renderHouses(houses) {
        const container = document.getElementById('gridContainer');
        document.getElementById('resultCount').innerText = `(${houses.length} căn)`;

        if (houses.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>🏠 Không tìm thấy căn nhà nào phù hợp tiêu chí.</p></div>';
            return;
        }

        container.innerHTML = houses.map(h => `
            <div class="house-card" id="card-${h.id}">
                <div class="card-header">
                    <div class="listed-price">${(h.Price || 0).toFixed(2)} Tỷ VNĐ</div>
                    <div class="address" title="${h.Address || ''}">📍 ${(h.Address || 'Chưa rõ').substring(0, 50)}...</div>
                </div>
                <div class="card-body">
                    <div>
                        <div class="specs">
                            <span class="spec-chip">📐 ${h.Area || 0} m²</span>
                            <span class="spec-chip">🛏️ ${h.Bedrooms || 0} PN</span>
                            <span class="spec-chip">🚿 ${h.Bathrooms || 0} WC</span>
                            <span class="spec-chip">🏢 ${h.Floors || 1}T</span>
                        </div>
                        <div class="details">
                            <p>🧭 Hướng: <b>${h['House direction'] || 'Chưa rõ'}</b></p>
                            <p>⚖️ Pháp lý: <b>${h['Legal status'] || 'Chưa rõ'}</b></p>
                        </div>
                    </div>
                    <div>
                        <button class="btn-evaluate" id="btn-${h.id}" onclick="evaluateHouse(${h.id})">
                            ✨ Thẩm Định Giá Bằng AI
                        </button>
                        <div class="ai-result-box" id="result-${h.id}"></div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async function evaluateHouse(houseId) {
        const btn = document.getElementById(`btn-${houseId}`);
        btn.innerText = "⏳ Đang tính...";
        btn.disabled = true;

        try {
            const res = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ house_id: houseId })
            });
            const data = await res.json();

            const box = document.getElementById(`result-${houseId}`);
            box.className = `ai-result-box ${data.status_class}`;
            box.style.display = 'block';

            box.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.7rem; font-weight:600;">🤖 AI Định Giá:</span>
                    <span class="ai-price">${data.ai_price.toFixed(2)} Tỷ</span>
                </div>
                <span class="ai-tag">${data.status_label}</span>
                <div class="ai-desc">
                    Lệch: <b>${data.diff_text}</b> (${data.diff_percent}%)<br>
                    ${data.comment}
                </div>
            `;
        } catch(e) {
            alert("Lỗi khi định giá!");
        } finally {
            btn.innerText = "🔄 Đã Thẩm Định";
            btn.disabled = false;
        }
    }

    triggerSearch();
</script>

</body>
</html>
"""


# ============================================================
# 5. API ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    """Trang chủ - Giao diện tìm kiếm và thẩm định giá nhà"""
    return HTML_CONTENT


@app.post("/api/search")
def search_houses(f: SearchFilter):
    """Tìm kiếm bất động sản theo tiêu chí"""
    if db_df.empty:
        return []

    filtered = db_df[
        (db_df["Area"] >= f.min_area) &
        (db_df["Area"] <= f.max_area) &
        (db_df["Price"] <= f.max_budget)
        ].copy()

    if f.address and f.address.strip():
        keyword = f.address.strip().lower()
        filtered = filtered[
            filtered["Clean_Address"].str.contains(keyword, na=False)
        ]

    if f.bedrooms > 0:
        filtered = filtered[
            (filtered["Bedrooms"] >= f.bedrooms - 1) &
            (filtered["Bedrooms"] <= f.bedrooms + 1)
            ]

    if f.floors > 0:
        filtered = filtered[
            (filtered["Floors"] >= f.floors - 1) &
            (filtered["Floors"] <= f.floors + 1)
            ]

    if f.house_direction != "Tất cả" and "House direction" in filtered.columns:
        filtered = filtered[
            filtered["House direction"].str.lower() == f.house_direction.lower()
            ]

    if f.legal_status != "Tất cả" and "Legal status" in filtered.columns:
        filtered = filtered[
            filtered["Legal status"].str.lower() == f.legal_status.lower()
            ]

    return filtered.head(30).fillna("Chưa rõ").to_dict(orient="records")


@app.post("/api/evaluate")
def evaluate_house(req: ValuationRequest):
    """Thẩm định giá nhà bằng AI"""
    if housing_model is None or housing_scaler is None:
        return {
            "ai_price": 0,
            "status_class": "fair",
            "status_label": "❌ LỖI MODEL",
            "diff_text": "0",
            "diff_percent": 0,
            "comment": "Model chưa được tải. Vui lòng kiểm tra file models/"
        }

    row = db_df.loc[db_df["id"] == req.house_id]
    if row.empty:
        return {
            "ai_price": 0,
            "status_class": "fair",
            "status_label": "❌ KHÔNG TÌM THẤY",
            "diff_text": "0",
            "diff_percent": 0,
            "comment": "Không tìm thấy căn nhà với ID này"
        }
    row = row.iloc[0]

    listed_price = float(row["Price"])

    # ==== PHẦN SỬA LỖI ====
    # Lấy 4 cột số để chuẩn hóa
    area = float(row["Area"]) if pd.notna(row["Area"]) else 50
    floors = float(row["Floors"]) if pd.notna(row["Floors"]) else 2
    bedrooms = float(row["Bedrooms"]) if pd.notna(row["Bedrooms"]) else 2
    bathrooms = float(row["Bathrooms"]) if pd.notna(row["Bathrooms"]) else 2

    # Mã hóa biến phân loại
    legal_encoded = 1 if str(row["Legal status"]) == "Have certificate" else 0
    furniture_encoded = 2 if str(row["Furniture state"]) == "Full" else 1 if str(
        row["Furniture state"]) == "Basic" else 0

    # Chuẩn hóa 4 cột số
    raw_numeric = np.array([[area, floors, bedrooms, bathrooms]])
    scaled_numeric = housing_scaler.transform(raw_numeric)

    # Kết hợp 4 cột đã chuẩn hóa + 2 cột đã mã hóa
    final_features = np.concatenate([scaled_numeric[0], [legal_encoded, furniture_encoded]]).reshape(1, -1)
    # ==== KẾT THÚC PHẦN SỬA LỖI ====

    try:
        ai_price = float(housing_model.predict(final_features)[0])
    except Exception as e:
        return {
            "ai_price": 0,
            "status_class": "fair",
            "status_label": "❌ LỖI DỰ ĐOÁN",
            "diff_text": "0",
            "diff_percent": 0,
            "comment": f"Lỗi: {str(e)[:50]}"
        }

    diff = listed_price - ai_price
    diff_percent = abs((diff / ai_price) * 100) if ai_price > 0 else 0

    if diff < -0.05 * ai_price:
        status_class = "cheap"
        status_label = "🔥 GIÁ HỜI (RẺ HƠN THỊ TRƯỜNG)"
        diff_text = f"Rẻ hơn {abs(diff):.2f} Tỷ"
        comment = "Mức giá rao bán thấp hơn định giá AI. Đây có thể là cơ hội tốt để đầu tư."
    elif diff > 0.05 * ai_price:
        status_class = "expensive"
        status_label = "⚠️ ĐỊNH GIÁ CAO (ĐẮT HƠN THỊ TRƯỜNG)"
        diff_text = f"Đắt hơn {diff:.2f} Tỷ"
        comment = "Mức giá rao bán cao hơn định giá AI. Có thể thương lượng để có giá tốt hơn."
    else:
        status_class = "fair"
        status_label = "✅ ĐÚNG GIÁ THỊ TRƯỜNG"
        diff_text = f"Sát giá ({abs(diff):.2f} Tỷ)"
        comment = "Mức giá phản ánh đúng giá trị trung bình theo mô hình AI."

    return {
        "ai_price": ai_price,
        "status_class": status_class,
        "status_label": status_label,
        "diff_text": diff_text,
        "diff_percent": round(diff_percent, 1),
        "comment": comment,
    }


@app.post("/api/predict")
def predict_price(req: PredictRequest):
    """Dự đoán giá nhà từ các đặc điểm đầu vào"""
    if housing_model is None or housing_scaler is None:
        return {
            "price": 0,
            "error": "Model chưa được tải"
        }

    # Mã hóa
    legal_encoded = 1 if req.legal_status == "Have certificate" else 0
    furniture_encoded = 2 if req.furniture_state == "Full" else 1 if req.furniture_state == "Basic" else 0

    # Chuẩn hóa 4 cột số
    raw_numeric = np.array([[req.area, req.floors, req.bedrooms, req.bathrooms]])
    scaled_numeric = housing_scaler.transform(raw_numeric)

    # Kết hợp
    final_features = np.concatenate([scaled_numeric[0], [legal_encoded, furniture_encoded]]).reshape(1, -1)

    try:
        price = float(housing_model.predict(final_features)[0])
        return {
            "price": price,
            "error": None
        }
    except Exception as e:
        return {
            "price": 0,
            "error": str(e)
        }


@app.get("/api/health")
def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "model_loaded": housing_model is not None,
        "data_loaded": not db_df.empty,
        "data_count": len(db_df)
    }


# ============================================================
# 6. CHẠY ỨNG DỤNG
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🏠 AI PORTAL - DỰ ĐOÁN GIÁ NHÀ")
    print("=" * 60)
    print(f"📌 Model loaded: {'✅ Có' if housing_model else '❌ Không'}")
    print(f"📌 Data loaded: {'✅ Có' if not db_df.empty else '❌ Không'} ({len(db_df)} căn)")
    print(f"📌 Truy cập Web: http://localhost:8002")
    print(f"📌 Tài liệu API: http://localhost:8002/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002)