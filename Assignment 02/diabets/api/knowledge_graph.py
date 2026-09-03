import requests
from bs4 import BeautifulSoup
import re
import json

class DiabetesKnowledgeGraph:
    """
    Knowledge Graph for diabetes advice in Vietnamese
    """
    
    def __init__(self):
        self.advice_cache = {}
        self.diabetes_info = self._load_diabetes_info()
        self.advice_rules = self._load_advice_rules()
    
    def _load_diabetes_info(self):
        """Load diabetes information in Vietnamese"""
        return {
            "type_1": {
                "description": "Tiểu đường type 1 là bệnh tự miễn, cơ thể không sản xuất insulin. Thường gặp ở trẻ em và người trẻ tuổi.",
                "symptoms": ["Khát nước nhiều", "Đi tiểu thường xuyên", "Đói nhiều", "Sụt cân không rõ nguyên nhân", "Mệt mỏi", "Nhìn mờ"],
                "management": ["Tiêm insulin hàng ngày", "Theo dõi đường huyết", "Đếm carbohydrate", "Chế độ ăn lành mạnh", "Tập thể dục đều đặn"]
            },
            "type_2": {
                "description": "Tiểu đường type 2 là tình trạng cơ thể kháng insulin hoặc không sản xuất đủ insulin. Đây là loại phổ biến nhất.",
                "symptoms": ["Khát nước nhiều", "Đi tiểu thường xuyên", "Đói nhiều", "Sụt cân", "Mệt mỏi", "Nhìn mờ", "Vết thương lâu lành", "Nhiễm trùng thường xuyên"],
                "management": ["Chế độ ăn lành mạnh", "Tập thể dục đều đặn", "Kiểm soát cân nặng", "Theo dõi đường huyết", "Thuốc uống", "Có thể tiêm insulin"]
            },
            "pre_diabetes": {
                "description": "Tiền tiểu đường là tình trạng đường huyết cao hơn bình thường nhưng chưa đủ để chẩn đoán tiểu đường type 2.",
                "symptoms": ["Thường không có triệu chứng", "Da sẫm màu ở một số vùng", "U nhỏ ở nách hoặc cổ"],
                "management": ["Giảm cân", "Chế độ ăn lành mạnh", "Tập thể dục đều đặn", "Theo dõi đường huyết", "Thay đổi lối sống"]
            },
            "gestational": {
                "description": "Tiểu đường thai kỳ phát triển trong thời kỳ mang thai và thường tự khỏi sau khi sinh.",
                "symptoms": ["Thường không có triệu chứng", "Khát nước nhiều", "Đi tiểu thường xuyên"],
                "management": ["Theo dõi đường huyết", "Chế độ ăn lành mạnh", "Tập thể dục đều đặn", "Có thể cần insulin"]
            }
        }
    
    def _load_advice_rules(self):
        """Load advice rules in Vietnamese"""
        return {
            "high_glucose": {
                "condition": lambda data: data.get("glucose_fasting", 0) > 126 or data.get("glucose_postprandial", 0) > 200,
                "advice": "🔴 Đường huyết của bạn đang ở mức cao. Vui lòng tham khảo ý kiến bác sĩ để được chẩn đoán và điều trị kịp thời. Theo dõi đường huyết thường xuyên và duy trì chế độ ăn uống lành mạnh."
            },
            "high_hba1c": {
                "condition": lambda data: data.get("hba1c", 0) > 6.5,
                "advice": "🔴 Chỉ số HbA1c của bạn cao hơn mức bình thường, cho thấy đường huyết không được kiểm soát tốt trong 2-3 tháng qua. Hãy đến gặp bác sĩ để điều chỉnh phác đồ điều trị."
            },
            "high_bmi": {
                "condition": lambda data: data.get("bmi", 0) > 25,
                "advice": "⚠️ Chỉ số BMI của bạn đang ở mức thừa cân. Thừa cân là yếu tố nguy cơ chính của tiểu đường type 2. Hãy xây dựng kế hoạch giảm cân với chế độ ăn uống lành mạnh và tập thể dục đều đặn."
            },
            "family_history": {
                "condition": lambda data: data.get("family_history_diabetes", 0) == 1,
                "advice": "📋 Gia đình bạn có người bị tiểu đường, điều này làm tăng nguy cơ mắc bệnh của bạn. Hãy kiểm tra sức khỏe định kỳ và duy trì lối sống lành mạnh."
            },
            "high_blood_pressure": {
                "condition": lambda data: data.get("systolic_bp", 0) > 130 or data.get("diastolic_bp", 0) > 85,
                "advice": "⚠️ Huyết áp của bạn đang ở mức cao. Huyết áp cao thường đi kèm với tiểu đường. Hãy theo dõi huyết áp thường xuyên và duy trì chế độ ăn ít muối."
            },
            "sedentary": {
                "condition": lambda data: data.get("physical_activity_minutes_per_week", 0) < 150,
                "advice": "🏃 Mức độ vận động của bạn còn thấp. Tập thể dục đều đặn (ít nhất 150 phút/tuần) giúp kiểm soát đường huyết và giảm nguy cơ tiểu đường."
            },
            "high_triglycerides": {
                "condition": lambda data: data.get("triglycerides", 0) > 150,
                "advice": "⚠️ Chỉ số Triglycerides của bạn cao. Triglycerides cao có liên quan đến nguy cơ tiểu đường. Hãy thay đổi chế độ ăn uống và tăng cường vận động."
            },
            "low_hdl": {
                "condition": lambda data: data.get("hdl_cholesterol", 0) < 40,
                "advice": "⚠️ HDL Cholesterol (cholesterol tốt) của bạn thấp. HDL thấp làm tăng nguy cơ bệnh tim mạch và tiểu đường. Hãy tập thể dục và ăn nhiều chất béo lành mạnh."
            },
            "high_ldl": {
                "condition": lambda data: data.get("ldl_cholesterol", 0) > 130,
                "advice": "⚠️ LDL Cholesterol (cholesterol xấu) của bạn cao. LDL cao là yếu tố nguy cơ của bệnh tim mạch và tiểu đường. Hãy điều chỉnh chế độ ăn uống và tham khảo ý kiến bác sĩ."
            }
        }
    
    def get_advice(self, patient_data, prediction_result):
        """Get personalized advice in Vietnamese"""
        advice = []
        
        # General advice based on prediction
        if prediction_result == 1:
            advice.append({
                "category": "🏥 Khuyến cáo y tế",
                "advice": "Bạn đang có nguy cơ cao hoặc được chẩn đoán mắc bệnh tiểu đường. Vui lòng đến cơ sở y tế để được khám và điều trị kịp thời.",
                "type": "medical"
            })
            
            # Determine likely type
            diabetes_type = self._determine_diabetes_type(patient_data)
            if diabetes_type:
                info = self.diabetes_info.get(diabetes_type, {})
                if info:
                    advice.append({
                        "category": f"📖 Thông tin về {diabetes_type.replace('_', ' ').title()}",
                        "advice": info.get("description", ""),
                        "type": "info"
                    })
                    if info.get("management"):
                        advice.append({
                            "category": "💊 Quản lý bệnh",
                            "advice": "Phương pháp quản lý: " + ", ".join(info.get("management", [])),
                            "type": "lifestyle"
                        })
        else:
            advice.append({
                "category": "✅ Phòng ngừa",
                "advice": "Bạn hiện chưa được chẩn đoán tiểu đường. Hãy duy trì lối sống lành mạnh để phòng ngừa bệnh tiểu đường.",
                "type": "prevention"
            })
        
        # Personalized advice based on risk factors
        for rule_name, rule in self.advice_rules.items():
            try:
                if rule["condition"](patient_data):
                    advice.append({
                        "category": "⚠️ Cảnh báo yếu tố nguy cơ",
                        "advice": rule["advice"],
                        "type": "risk"
                    })
            except:
                continue
        
        # General health advice
        advice.append({
            "category": "💪 Lời khuyên sức khỏe tổng quát",
            "advice": "Duy trì chế độ ăn cân bằng với nhiều rau xanh, trái cây, ngũ cốc nguyên hạt. Hạn chế đồ ăn chế biến sẵn và đồ uống có đường. Uống đủ nước và ngủ đủ giấc.",
            "type": "lifestyle"
        })
        
        # Additional diabetes prevention advice
        advice.append({
            "category": "📋 Theo dõi sức khỏe định kỳ",
            "advice": "Khám sức khỏe định kỳ 6 tháng/lần để phát hiện sớm các vấn đề sức khỏe. Đặc biệt quan trọng đối với người có yếu tố nguy cơ tiểu đường.",
            "type": "prevention"
        })
        
        return advice
    
    def _determine_diabetes_type(self, data):
        """Determine possible diabetes type based on patient data"""
        age = data.get("age", 0)
        bmi = data.get("bmi", 0)
        
        if age < 20:
            return "type_1"
        elif age >= 20 and bmi > 25:
            return "type_2"
        elif age >= 20 and bmi <= 25:
            return "type_2"
        return "type_2"