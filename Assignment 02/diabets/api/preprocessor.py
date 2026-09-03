import pandas as pd
import numpy as np
import joblib
import os

class DiabetesPreprocessor:
    def __init__(self, model_path="../model/diabetes_model_pipeline.joblib"):
        self.model_path = model_path
        self.pipeline = None
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load the saved pipeline and metadata"""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load metadata
        metadata_path = self.model_path.replace("diabetes_model_pipeline.joblib", "model_metadata.joblib")
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
    
    def predict(self, input_data):
        """
        Predict diabetes diagnosis from input data
        
        Args:
            input_data: dict or list of dicts with patient features
        
        Returns:
            dict: prediction results
        """
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)
        
        # Ensure all required columns exist
        required_columns = self.metadata.get("numerical_features", []) + self.metadata.get("categorical_features", [])
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Make prediction
        prediction = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)
        
        # Return results
        results = []
        for i in range(len(df)):
            results.append({
                "prediction": int(prediction[i]),
                "probability": float(probabilities[i][1]),
                "confidence": float(max(probabilities[i])),
                "class": "Diabetic" if prediction[i] == 1 else "Non-Diabetic"
            })
        
        return results if len(results) > 1 else results[0]
    
    def get_feature_info(self):
        """Get information about features used by the model"""
        if self.metadata:
            return {
                "numerical_features": self.metadata.get("numerical_features", []),
                "categorical_features": self.metadata.get("categorical_features", []),
                "target": self.metadata.get("target", "diagnosed_diabetes"),
                "model": self.metadata.get("selected_model", "Random Forest")
            }
        return {}