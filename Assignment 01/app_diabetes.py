# -*- coding: utf-8 -*-
"""
Ứng dụng Dự đoán Tiểu đường
AI Portal - Diabetes Prediction System
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
    title="AI Portal - Dự Đoán Tiểu Đường",
    description="Ứng dụng dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số sức khỏe",
    version="1.0.0"
)

# ============================================================
# 2. TẢI MODEL
# ============================================================

diabetes_pipeline = None

try:
    diabetes_pipeline = joblib.load("models/diabetes_model_pipeline.pkl")
    print("✅ Đã tải model dự đoán tiểu đường thành công!")
except FileNotFoundError:
    print("❌ Không tìm thấy file model. Vui lòng chạy notebook để train model trước.")
    print("   Đường dẫn cần có: models/diabetes_model_pipeline.pkl")
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")


# ============================================================
# 3. ĐỊNH NGHĨA REQUEST MODEL (8 đặc trưng)
# ============================================================

class DiabetesRequest(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float  # ← ĐÃ THÊM
    age: float


# ============================================================
# 4. HTML CONTENT (Đã cập nhật)
# ============================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Dự Đoán Tiểu Đường - AI Portal</title>
    <style>
        :root { --primary: #2563eb; --hover: #1d4ed8; --bg: #f0f4f8; --card: #ffffff; --text: #0f172a; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }

        .container { max-width: 720px; width: 100%; background: var(--card); border-radius: 16px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }

        .header { text-align: center; margin-bottom: 24px; }
        .header h1 { font-size: 1.5rem; color: var(--primary); font-weight: 700; }
        .header p { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
        .header .badge { display: inline-block; background: #dbeafe; color: #1d4ed8; padding: 2px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; margin-top: 8px; }

        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        @media(max-width: 500px) { .form-grid { grid-template-columns: 1fr; } }

        .form-group { display: flex; flex-direction: column; gap: 4px; }
        .form-group.full-width { grid-column: 1 / -1; }

        label { font-size: 0.78rem; font-weight: 600; color: #475569; }
        label .unit { font-weight: 400; color: #94a3b8; }

        input { padding: 10px 12px; border: 1.5px solid #e2e8f0; border-radius: 8px; font-size: 0.9rem; outline: none; transition: border-color 0.2s; background: #f8fafc; }
        input:focus { border-color: var(--primary); background: #fff; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }

        .btn-predict { width: 100%; padding: 14px; background: var(--primary); color: #fff; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 16px; transition: background 0.2s; }
        .btn-predict:hover { background: var(--hover); }
        .btn-predict:disabled { opacity: 0.6; cursor: not-allowed; }

        .result-box { margin-top: 20px; padding: 20px; border-radius: 10px; display: none; }
        .result-box.positive { display: block; background: #fef2f2; border: 1px solid #fca5a5; }
        .result-box.negative { display: block; background: #ecfdf5; border: 1px solid #86efac; }
        .result-box.error { display: block; background: #fffbeb; border: 1px solid #fcd34d; }

        .result-title { font-size: 1.2rem; font-weight: 700; }
        .result-detail { font-size: 0.9rem; margin-top: 6px; color: #475569; }
        .result-probability { margin-top: 8px; }
        .result-risk { display: inline-block; padding: 2px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
        .risk-high { background: #fee2e2; color: #b91c1c; }
        .risk-medium { background: #fef3c7; color: #92400e; }
        .risk-low { background: #dcfce7; color: #15803d; }

        .footer { text-align: center; margin-top: 20px; font-size: 0.7rem; color: #94a3b8; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🩺 Dự Đoán Tiểu Đường</h1>
        <p>Nhập các chỉ số sức khỏe để đánh giá nguy cơ mắc bệnh tiểu đường</p>
        <span class="badge">⚕️ Mô hình Random Forest • Độ chính xác 86.36%</span>
    </div>

    <form id="diabetesForm">
        <div class="form-grid">
            <div class="form-group">
                <label>Số lần mang thai</label>
                <input type="number" id="pregnancies" value="0" min="0" step="1" required>
            </div>
            <div class="form-group">
                <label>Glucose <span class="unit">(mg/dL)</span></label>
                <input type="number" id="glucose" value="100" min="0" step="1" required>
            </div>
            <div class="form-group">
                <label>Huyết áp <span class="unit">(mmHg)</span></label>
                <input type="number" id="blood_pressure" value="70" min="0" step="1" required>
            </div>
            <div class="form-group">
                <label>Độ dày da <span class="unit">(mm)</span></label>
                <input type="number" id="skin_thickness" value="25" min="0" step="1" required>
            </div>
            <div class="form-group">
                <label>Insulin <span class="unit">(µU/mL)</span></label>
                <input type="number" id="insulin" value="80" min="0" step="1" required>
            </div>
            <div class="form-group">
                <label>BMI</label>
                <input type="number" id="bmi" value="28.0" min="0" step="0.1" required>
            </div>
            <div class="form-group">
                <label>Diabetes Pedigree</label>
                <input type="number" id="diabetes_pedigree_function" value="0.5" min="0" step="0.01" required>
            </div>
            <div class="form-group">
                <label>Tuổi</label>
                <input type="number" id="age" value="30" min="0" step="1" required>
            </div>
        </div>
        <button type="submit" class="btn-predict" id="btnPredict">🔍 Dự Đoán Nguy Cơ</button>
    </form>

    <div class="result-box" id="resultBox">
        <div class="result-title" id="resultTitle"></div>
        <div class="result-detail" id="resultDetail"></div>
        <div class="result-probability" id="resultProbability"></div>
        <div class="result-risk" id="resultRisk"></div>
    </div>

    <div class="footer">
        <span>⚠️ Thông tin mang tính tham khảo, không thay thế chẩn đoán y tế</span>
    </div>
</div>

<script>
    document.getElementById('diabetesForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('btnPredict');
        btn.innerText = "⏳ Đang phân tích...";
        btn.disabled = true;
        document.getElementById('resultBox').className = 'result-box';
        document.getElementById('resultBox').style.display = 'none';

        const payload = {
            pregnancies: parseFloat(document.getElementById('pregnancies').value),
            glucose: parseFloat(document.getElementById('glucose').value),
            blood_pressure: parseFloat(document.getElementById('blood_pressure').value),
            skin_thickness: parseFloat(document.getElementById('skin_thickness').value),
            insulin: parseFloat(document.getElementById('insulin').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            diabetes_pedigree_function: parseFloat(document.getElementById('diabetes_pedigree_function').value),
            age: parseFloat(document.getElementById('age').value)
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            const box = document.getElementById('resultBox');
            box.className = 'result-box ' + (data.status_class || '');
            box.style.display = 'block';

            document.getElementById('resultTitle').textContent = data.prediction || 'Không có kết quả';
            document.getElementById('resultDetail').textContent = data.detail || '';

            const probEl = document.getElementById('resultProbability');
            if (data.probability !== undefined) {
                probEl.textContent = `Xác suất: ${(data.probability * 100).toFixed(2)}%`;
            }

            const riskEl = document.getElementById('resultRisk');
            if (data.risk_level) {
                const riskMap = {
                    'Cao': 'risk-high',
                    'Trung bình': 'risk-medium',
                    'Thấp': 'risk-low'
                };
                riskEl.className = 'result-risk ' + (riskMap[data.risk_level] || '');
                riskEl.textContent = `Mức độ rủi ro: ${data.risk_level}`;
            }

        } catch (error) {
            const box = document.getElementById('resultBox');
            box.className = 'result-box error';
            box.style.display = 'block';
            document.getElementById('resultTitle').textContent = '❌ Lỗi kết nối';
            document.getElementById('resultDetail').textContent = 'Không thể kết nối đến máy chủ. Vui lòng thử lại.';
        } finally {
            btn.innerText = "🔍 Dự Đoán Nguy Cơ";
            btn.disabled = false;
        }
    });
</script>

</body>
</html>
"""


# ============================================================
# 5. API ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    """Trang chủ - Giao diện dự đoán tiểu đường"""
    return HTML_CONTENT


@app.post("/api/predict")
def predict_diabetes(req: DiabetesRequest):
    """
    Dự đoán nguy cơ tiểu đường từ các chỉ số sức khỏe

    - **pregnancies**: Số lần mang thai
    - **glucose**: Nồng độ glucose (mg/dL)
    - **blood_pressure**: Huyết áp tâm trương (mmHg)
    - **skin_thickness**: Độ dày nếp da (mm)
    - **insulin**: Nồng độ insulin (µU/mL)
    - **bmi**: Chỉ số khối cơ thể
    - **diabetes_pedigree_function**: Chỉ số di truyền tiểu đường
    - **age**: Tuổi
    """
    if diabetes_pipeline is None:
        return {
            "prediction": "❌ LỖI HỆ THỐNG",
            "detail": "Model chưa được tải. Vui lòng kiểm tra file models/diabetes_model_pipeline.pkl",
            "probability": 0,
            "risk_level": "Không xác định",
            "status_class": "error"
        }

    # Tạo feature vector với ĐÚNG 8 đặc trưng
    raw_features = np.array([[
        req.pregnancies,
        req.glucose,
        req.blood_pressure,
        req.skin_thickness,
        req.insulin,
        req.bmi,
        req.diabetes_pedigree_function,
        req.age
    ]])

    try:
        # Dự đoán
        prediction = diabetes_pipeline.predict(raw_features)
        probability = diabetes_pipeline.predict_proba(raw_features)[0][1]

        # Xác định kết quả
        if prediction[0] == 1:
            result = "⚠️ Có nguy cơ mắc bệnh tiểu đường"
            status_class = "positive"
        else:
            result = "✅ Không có nguy cơ mắc bệnh tiểu đường"
            status_class = "negative"

        # Mức độ rủi ro
        if probability > 0.7:
            risk_level = "Cao"
        elif probability > 0.4:
            risk_level = "Trung bình"
        else:
            risk_level = "Thấp"

        # Lời khuyên
        if prediction[0] == 1:
            detail = "Bạn nên tham khảo ý kiến bác sĩ để được tư vấn và kiểm tra chuyên sâu."
        else:
            detail = "Duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ."

        return {
            "prediction": result,
            "detail": detail,
            "probability": probability,
            "risk_level": risk_level,
            "status_class": status_class
        }

    except Exception as e:
        return {
            "prediction": f"❌ Lỗi dự đoán",
            "detail": str(e),
            "probability": 0,
            "risk_level": "Không xác định",
            "status_class": "error"
        }


@app.get("/api/health")
def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "model_loaded": diabetes_pipeline is not None,
        "message": "API đang hoạt động bình thường"
    }


# ============================================================
# 6. CHẠY ỨNG DỤNG
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🩺 AI PORTAL - DỰ ĐOÁN TIỂU ĐƯỜNG")
    print("=" * 60)
    print(f"📌 Model loaded: {'✅ Có' if diabetes_pipeline else '❌ Không'}")
    print(f"📌 Truy cập Web: http://localhost:8001")
    print(f"📌 Tài liệu API: http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)