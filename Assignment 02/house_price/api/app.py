"""
Ứng dụng dự đoán giá nhà - Hợp nhất API và Web Server
Chạy trên cổng 8002, giao diện responsive cho mobile
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os

# === QUAN TRỌNG: Thiết lập đường dẫn đúng ===
# Lấy đường dẫn tuyệt đối đến thư mục chứa file app.py
API_DIR = Path(__file__).resolve().parent          # .../house_price/api
PROJECT_DIR = API_DIR.parent                      # .../house_price
WEB_DIR = PROJECT_DIR / "web"                    # .../house_price/web

# Khởi tạo Flask với đường dẫn templates và static từ thư mục web
app = Flask(
    __name__,
    template_folder=str(WEB_DIR / 'templates'),   # web/templates/
    static_folder=str(WEB_DIR / 'static')         # web/static/
)
CORS(app)

print("=" * 50)
print("📁 PROJECT STRUCTURE")
print("=" * 50)
print(f"📁 API_DIR: {API_DIR}")
print(f"📁 PROJECT_DIR: {PROJECT_DIR}")
print(f"📁 WEB_DIR: {WEB_DIR}")
print(f"📁 Templates: {WEB_DIR / 'templates'}")
print(f"📁 Static: {WEB_DIR / 'static'}")

# Kiểm tra templates folder
templates_path = WEB_DIR / 'templates'
if templates_path.exists():
    print("✅ Templates folder exists")
    print(f"   Files: {[f.name for f in templates_path.iterdir()]}")
else:
    print("❌ Templates folder NOT found!")
    print(f"   Please create: {templates_path}")

# Kiểm tra static folder
static_path = WEB_DIR / 'static'
if static_path.exists():
    print("✅ Static folder exists")
else:
    print("❌ Static folder NOT found!")
    print(f"   Please create: {static_path}")
print("=" * 50)

# Đường dẫn đến model
MODEL_DIR = PROJECT_DIR / "model"

# Load model và preprocessor
pipeline_path = MODEL_DIR / "model_pipeline.joblib"
config_path = MODEL_DIR / "model_config.json"

try:
    pipeline = joblib.load(pipeline_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print("✅ Model loaded successfully!")
    print(f"📊 Model: {config.get('selected_model', 'Unknown')}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    pipeline = None
    config = {
        "target_unit": "million VND",
        "numeric_features": ["timeline_hours", "area_m2", "bedrooms", "bathrooms", "floors"],
        "categorical_features": ["location"]
    }

# === Knowledge Graph Class (inline) ===
class PriceFactors:
    LOCATION_FACTORS = {
        "Gò Vấp, Hồ Chí Minh": 0.85,
        "Tân Phú, Hồ Chí Minh": 0.80,
        "Đống Đa, Hà Nội": 1.40,
        "Bình Tân, Hồ Chí Minh": 0.75,
        "Long Biên, Hà Nội": 1.20,
        "Quận 12, Hồ Chí Minh": 0.95,
        "Cầu Giấy, Hà Nội": 1.60,
        "Tân Bình, Hồ Chí Minh": 0.90,
        "Thanh Xuân, Hà Nội": 1.25,
        "Hoàng Mai, Hà Nội": 0.85,
        "Hà Đông, Hà Nội": 0.95,
        "Quận 8, Hồ Chí Minh": 0.80,
        "Thủ Đức, Hồ Chí Minh": 0.90,
        "Bình Thạnh, Hồ Chí Minh": 1.00,
        "Hai Bà Trưng, Hà Nội": 1.50
    }
    
    @classmethod
    def calculate_factors(cls, location, area_m2, bedrooms, bathrooms, floors, frontage, prediction):
        location_factor = cls.LOCATION_FACTORS.get(location, 1.0)
        area_factor = min(area_m2 / 50, 2.0) if area_m2 > 0 else 1.0
        bedroom_factor = 0.8 + (bedrooms - 2) * 0.15 if bedrooms >= 2 else 0.8
        bathroom_factor = 0.8 + (bathrooms - 1) * 0.15 if bathrooms >= 1 else 0.8
        floor_factor = 0.8 + (floors - 1) * 0.1 if floors >= 1 else 0.8
        frontage_factor = 1.15 if frontage else 1.0
        
        total_factor = location_factor * area_factor * bedroom_factor * bathroom_factor * floor_factor * frontage_factor
        base_price = prediction / total_factor if total_factor > 0 else prediction
        
        explanations = []
        if location_factor > 1.2:
            explanations.append(f"📍 Vị trí {location} có giá cao hơn trung bình (hệ số {location_factor:.2f})")
        elif location_factor < 0.8:
            explanations.append(f"📍 Vị trí {location} có giá thấp hơn trung bình (hệ số {location_factor:.2f})")
        else:
            explanations.append(f"📍 Vị trí {location} có giá ở mức trung bình")
        
        if area_m2 > 100:
            explanations.append(f"📐 Diện tích {area_m2}m² lớn, tăng giá (hệ số {area_factor:.2f})")
        elif area_m2 < 40:
            explanations.append(f"📐 Diện tích {area_m2}m² nhỏ, giảm giá (hệ số {area_factor:.2f})")
        else:
            explanations.append(f"📐 Diện tích {area_m2}m² ở mức trung bình")
        
        if bedrooms >= 4:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình lớn (hệ số {bedroom_factor:.2f})")
        elif bedrooms <= 2:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình nhỏ (hệ số {bedroom_factor:.2f})")
        else:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình trung bình (hệ số {bedroom_factor:.2f})")
        
        if bathrooms >= 3:
            explanations.append(f"🛁 {bathrooms} phòng tắm - tiện nghi cao (hệ số {bathroom_factor:.2f})")
        else:
            explanations.append(f"🛁 {bathrooms} phòng tắm (hệ số {bathroom_factor:.2f})")
        
        if floors >= 5:
            explanations.append(f"🏗️ {floors} tầng - nhà cao tầng (hệ số {floor_factor:.2f})")
        else:
            explanations.append(f"🏗️ {floors} tầng (hệ số {floor_factor:.2f})")
        
        if frontage:
            explanations.append("🚪 Có mặt tiền - tăng giá trị (hệ số 1.15)")
        else:
            explanations.append("🚪 Không có mặt tiền")
        
        explanations.append(f"💰 Giá dự đoán: {prediction:,.0f} triệu VND")
        
        return {
            "base_price": round(base_price, 2),
            "location_factor": round(location_factor, 2),
            "area_factor": round(area_factor, 2),
            "bedroom_factor": round(bedroom_factor, 2),
            "bathroom_factor": round(bathroom_factor, 2),
            "floor_factor": round(floor_factor, 2),
            "frontage_factor": round(frontage_factor, 2),
            "total_factor": round(total_factor, 2),
            "explanation": " | ".join(explanations)
        }

class HousePriceKnowledgeGraph:
    def __init__(self):
        self.graph = {"nodes": [], "edges": [], "relationships": {}}
    
    def create_graph(self, location, area_m2, bedrooms, bathrooms, floors, frontage, prediction, factors):
        self.graph = {"nodes": [], "edges": [], "relationships": {}}
        
        self._add_node("house", "🏠 Nhà", "Chính", {
            "Vị trí": location,
            "Diện tích": f"{area_m2}m²",
            "Phòng ngủ": f"{bedrooms} phòng",
            "Phòng tắm": f"{bathrooms} phòng",
            "Số tầng": f"{floors} tầng",
            "Mặt tiền": "Có" if frontage else "Không",
            "Giá dự đoán": f"{prediction:,.0f} triệu VND"
        })
        
        self._add_node("location", "📍 Vị trí", "Yếu tố", {
            "Tên": location,
            "Hệ số": factors.get("location_factor", 1.0)
        })
        self._add_node("area", "📐 Diện tích", "Yếu tố", {
            "Giá trị": f"{area_m2}m²",
            "Hệ số": factors.get("area_factor", 1.0)
        })
        self._add_node("bedrooms", "🛏️ Phòng ngủ", "Yếu tố", {
            "Số phòng": bedrooms,
            "Hệ số": factors.get("bedroom_factor", 1.0)
        })
        self._add_node("bathrooms", "🛁 Phòng tắm", "Yếu tố", {
            "Số phòng": bathrooms,
            "Hệ số": factors.get("bathroom_factor", 1.0)
        })
        self._add_node("floors", "🏗️ Số tầng", "Yếu tố", {
            "Số tầng": floors,
            "Hệ số": factors.get("floor_factor", 1.0)
        })
        self._add_node("frontage", "🚪 Mặt tiền", "Yếu tố", {
            "Có mặt tiền": "Có" if frontage else "Không",
            "Hệ số": factors.get("frontage_factor", 1.0)
        })
        self._add_node("price", "💰 Giá", "Kết quả", {
            "Giá trị": f"{prediction:,.0f} triệu VND"
        })
        
        self._add_edge("location", "house", "ảnh hưởng", factors.get("location_factor", 1.0))
        self._add_edge("area", "house", "ảnh hưởng", factors.get("area_factor", 1.0))
        self._add_edge("bedrooms", "house", "ảnh hưởng", factors.get("bedroom_factor", 1.0))
        self._add_edge("bathrooms", "house", "ảnh hưởng", factors.get("bathroom_factor", 1.0))
        self._add_edge("floors", "house", "ảnh hưởng", factors.get("floor_factor", 1.0))
        self._add_edge("frontage", "house", "ảnh hưởng", factors.get("frontage_factor", 1.0))
        self._add_edge("house", "price", "dẫn đến", 1.0)
        
        return self.graph
    
    def _add_node(self, node_id, label, node_type, properties):
        self.graph["nodes"].append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "properties": properties
        })
    
    def _add_edge(self, source, target, label, weight):
        self.graph["edges"].append({
            "source": source,
            "target": target,
            "label": label,
            "weight": weight
        })

# Khởi tạo Knowledge Graph
kg = HousePriceKnowledgeGraph()

# Danh sách location
COMMON_LOCATIONS = [
    "Gò Vấp, Hồ Chí Minh",
    "Tân Phú, Hồ Chí Minh",
    "Đống Đa, Hà Nội",
    "Bình Tân, Hồ Chí Minh",
    "Long Biên, Hà Nội",
    "Quận 12, Hồ Chí Minh",
    "Cầu Giấy, Hà Nội",
    "Tân Bình, Hồ Chí Minh",
    "Thanh Xuân, Hà Nội",
    "Hoàng Mai, Hà Nội",
    "Hà Đông, Hà Nội",
    "Quận 8, Hồ Chí Minh",
    "Thủ Đức, Hồ Chí Minh",
    "Bình Thạnh, Hồ Chí Minh",
    "Hai Bà Trưng, Hà Nội"
]


@app.route('/')
def index():
    """Trang chủ - Giao diện web responsive"""
    try:
        return render_template('index.html', locations=COMMON_LOCATIONS)
    except Exception as e:
        return f"❌ Lỗi render template: {e}", 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """API dự đoán giá nhà"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        # Validate input
        required_fields = ['location', 'area_m2', 'bedrooms', 'bathrooms', 'floors']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Chuyển đổi dữ liệu
        input_data = {
            'location': [data.get('location', '')],
            'timeline_hours': [float(data.get('timeline_hours', 0))],
            'area_m2': [float(data.get('area_m2', 0))],
            'bedrooms': [float(data.get('bedrooms', 0))],
            'bathrooms': [float(data.get('bathrooms', 0))],
            'floors': [float(data.get('floors', 0))],
            'frontage': [data.get('frontage', False)]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Dự đoán
        if pipeline is not None:
            prediction = float(pipeline.predict(input_df)[0])
        else:
            # Fallback
            prediction = (
                data.get('area_m2', 50) * 150 + 
                data.get('bedrooms', 3) * 400 + 
                data.get('bathrooms', 2) * 300 +
                data.get('floors', 1) * 200
            )
            if data.get('frontage', False):
                prediction *= 1.1
        
        prediction = max(prediction, 0)
        
        # Tính toán yếu tố ảnh hưởng
        factors = PriceFactors.calculate_factors(
            location=data.get('location', ''),
            area_m2=data.get('area_m2', 50),
            bedrooms=data.get('bedrooms', 3),
            bathrooms=data.get('bathrooms', 2),
            floors=data.get('floors', 1),
            frontage=data.get('frontage', False),
            prediction=prediction
        )
        
        # Tạo Knowledge Graph
        kg_data = kg.create_graph(
            location=data.get('location', ''),
            area_m2=data.get('area_m2', 50),
            bedrooms=data.get('bedrooms', 3),
            bathrooms=data.get('bathrooms', 2),
            floors=data.get('floors', 1),
            frontage=data.get('frontage', False),
            prediction=prediction,
            factors=factors
        )
        
        # Định dạng giá
        million = prediction
        billion = prediction / 1000
        
        formatted_million = f"{million:,.0f} triệu VND"
        if billion >= 1:
            formatted_full = f"{billion:,.2f} tỷ VND"
        else:
            formatted_full = formatted_million
        
        response = {
            "success": True,
            "prediction": prediction,
            "formatted_price": formatted_million,
            "formatted_price_full": formatted_full,
            "factors": factors,
            "knowledge_graph": kg_data,
            "input_summary": {
                "location": data.get('location', ''),
                "area_m2": data.get('area_m2', 0),
                "bedrooms": data.get('bedrooms', 0),
                "bathrooms": data.get('bathrooms', 0),
                "floors": data.get('floors', 0),
                "frontage": "Có" if data.get('frontage', False) else "Không"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Lấy danh sách location"""
    return jsonify({
        "success": True,
        "locations": COMMON_LOCATIONS,
        "count": len(COMMON_LOCATIONS)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra trạng thái"""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "unit": "million VND"
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🏠 HOUSE PRICE PREDICTION APP")
    print("=" * 50)
    print(f"📁 API_DIR: {API_DIR}")
    print(f"📁 WEB_DIR: {WEB_DIR}")
    print(f"📁 Templates: {WEB_DIR / 'templates'}")
    print(f"📡 Server: http://localhost:8002")
    print(f"📱 Mobile: http://<YOUR_IP>:8002")
    print(f"📊 Unit: triệu VND (1,000 triệu = 1 tỷ VND)")
    print("=" * 50)
    print("✅ Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=8002)