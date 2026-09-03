from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
from preprocessor import DiabetesPreprocessor
from knowledge_graph import DiabetesKnowledgeGraph

app = Flask(__name__, 
            static_folder='../web/static',
            template_folder='../web/templates')
CORS(app)

# Initialize components
try:
    preprocessor = DiabetesPreprocessor("../model/diabetes_model_pipeline.joblib")
except Exception as e:
    print(f"Error loading model: {e}")
    preprocessor = None

knowledge_graph = DiabetesKnowledgeGraph()

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": preprocessor is not None,
        "service": "Diabetes Prediction API"
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get information about required features"""
    if preprocessor:
        feature_info = preprocessor.get_feature_info()
        return jsonify(feature_info)
    return jsonify({"error": "Model not loaded"}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict diabetes diagnosis from patient data"""
    if not preprocessor:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = preprocessor.predict(data)
        
        if isinstance(data, dict):
            patient_data = data
        else:
            patient_data = data[0] if data else {}
        
        advice = knowledge_graph.get_advice(patient_data, result["prediction"])
        
        return jsonify({
            "success": True,
            "prediction": result,
            "advice": advice,
            "message": f"Patient is classified as {result['class']} with {result['probability']*100:.1f}% probability"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict diabetes diagnosis for multiple patients"""
    if not preprocessor:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Expected list of patient data"}), 400
        
        results = preprocessor.predict(data)
        
        for i, result in enumerate(results):
            patient_data = data[i] if i < len(data) else {}
            advice = knowledge_graph.get_advice(patient_data, result["prediction"])
            result["advice"] = advice
        
        return jsonify({
            "success": True,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/advice', methods=['POST'])
def get_advice():
    """Get personalized advice without making a prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if preprocessor:
            result = preprocessor.predict(data)
            prediction = result["prediction"]
        else:
            prediction = 0
        
        advice = knowledge_graph.get_advice(data, prediction)
        
        return jsonify({
            "success": True,
            "advice": advice
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/info/diabetes')
def get_diabetes_info():
    """Get general information about diabetes"""
    return jsonify({
        "success": True,
        "info": {
            "what_is_diabetes": "Diabetes is a chronic condition that affects how your body turns food into energy.",
            "types": {
                "type_1": "Type 1 diabetes is an autoimmune condition. The body doesn't produce insulin.",
                "type_2": "Type 2 diabetes is the most common type. The body doesn't use insulin properly.",
                "gestational": "Gestational diabetes develops during pregnancy.",
                "pre_diabetes": "Prediabetes means blood sugar levels are higher than normal."
            },
            "risk_factors": [
                "Family history of diabetes",
                "Overweight or obesity",
                "Physical inactivity",
                "Age (45 or older)",
                "High blood pressure"
            ],
            "prevention": [
                "Maintain healthy weight",
                "Eat balanced diet",
                "Exercise regularly",
                "Monitor blood sugar"
            ]
        }
    })

if __name__ == '__main__':
    # Chạy trên cổng 8002 cho mobile
    app.run(debug=True, host='0.0.0.0', port=8002)