# api/app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import sys
from knowledge_graph import ProductKnowledgeGraph

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
            template_folder='../web',
            static_folder='../web',
            static_url_path='')

CORS(app)

# Đường dẫn đến model
MODEL_DIR = Path(__file__).parent.parent / "model"

# Load model và preprocessor
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
model = joblib.load(MODEL_DIR / "model.joblib")

# Load config
with open(MODEL_DIR / "model_config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

# Danh sách các class
CLASSES = [
    "Blouses", "Dresses", "Fine gauge", "Intimates", "Jackets",
    "Jeans", "Knits", "Layering", "Legwear", "Lounge",
    "Outerwear", "Pants", "Shorts", "Skirts", "Sleep",
    "Sweaters", "Swim", "Trend", "Other"
]

# Khởi tạo Knowledge Graph
kg = ProductKnowledgeGraph()

@app.route('/')
def index():
    """Trang chủ"""
    return send_from_directory('../web', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint dự đoán"""
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        
        # Tạo DataFrame từ input
        input_data = {
            'Age': [float(data.get('age', 0))],
            'Rating': [float(data.get('rating', 0))],
            'Recommended IND': [int(data.get('recommended_ind', 0))],
            'Positive Feedback Count': [int(data.get('feedback_count', 0))],
            'review_length': [len(data.get('review_text', ''))],
            'review_word_count': [len(data.get('review_text', '').split())],
            'Division Name': [data.get('division_name', 'General')],
            'Department Name': [data.get('department_name', 'Tops')],
            'Review Text': [data.get('review_text', '')]
        }
        
        df = pd.DataFrame(input_data)
        
        # Preprocess
        X_transformed = preprocessor.transform(df)
        
        # Dự đoán
        prediction = model.predict(X_transformed)[0]
        
        # Xác suất nếu có
        probability = None
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_transformed)[0]
            probability = float(max(probabilities))
        
        # Lấy top 5 predictions
        top_predictions = []
        if probabilities is not None:
            top_indices = np.argsort(probabilities)[::-1][:5]
            for idx in top_indices:
                top_predictions.append({
                    'class': CLASSES[idx],
                    'probability': float(probabilities[idx])
                })
        
        # Lấy thông tin sản phẩm từ Knowledge Graph
        product_info = kg.get_product_info(prediction)
        recommendations = kg.recommend_products(prediction)
        
        # Lấy knowledge graph data
        kg_data = kg.get_knowledge_graph_data(prediction)
        
        response = {
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'top_predictions': top_predictions,
            'product_info': product_info,
            'recommendations': recommendations,
            'kg_data': kg_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/product/<product_name>')
def get_product_info(product_name):
    """Lấy thông tin sản phẩm từ Knowledge Graph"""
    info = kg.get_product_info(product_name)
    if info:
        return jsonify({
            'success': True,
            'product_info': info
        })
    return jsonify({
        'success': False,
        'error': 'Product not found'
    }), 404

@app.route('/recommend/<product_name>')
def get_recommendations(product_name):
    """Lấy gợi ý sản phẩm"""
    recommendations = kg.recommend_products(product_name)
    return jsonify({
        'success': True,
        'product': product_name,
        'recommendations': recommendations
    })

@app.route('/products')
def list_products():
    """Danh sách tất cả sản phẩm"""
    products = kg.get_all_products()
    return jsonify({
        'success': True,
        'products': products
    })

@app.route('/kg/<product_name>')
def get_knowledge_graph(product_name):
    """Lấy dữ liệu knowledge graph cho một sản phẩm"""
    kg_data = kg.get_knowledge_graph_data(product_name)
    if kg_data:
        return jsonify({
            'success': True,
            'nodes': kg_data['nodes'],
            'edges': kg_data['edges']
        })
    return jsonify({
        'success': False,
        'error': 'Product not found'
    }), 404

@app.route('/compare')
def compare_products():
    """So sánh hai sản phẩm"""
    p1 = request.args.get('p1')
    p2 = request.args.get('p2')
    
    if not p1 or not p2:
        return jsonify({
            'success': False,
            'error': 'Missing product names'
        }), 400
    
    common = kg.find_common_interests(p1, p2)
    info1 = kg.get_product_info(p1)
    info2 = kg.get_product_info(p2)
    
    return jsonify({
        'success': True,
        'product1': info1,
        'product2': info2,
        'common_interests': common
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8002)