"""
Knowledge Graph cho ứng dụng dự đoán giá nhà
Giải thích các yếu tố ảnh hưởng đến giá nhà
"""

class PriceFactors:
    """Tính toán các yếu tố ảnh hưởng đến giá nhà"""
    
    # Hệ số ảnh hưởng của location (dựa trên dữ liệu thực tế)
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
        """Tính toán các yếu tố ảnh hưởng"""
        
        # Hệ số location
        location_factor = cls.LOCATION_FACTORS.get(location, 1.0)
        
        # Hệ số diện tích (diện tích càng lớn giá càng cao)
        if area_m2 > 0:
            area_factor = min(area_m2 / 50, 2.0)
        else:
            area_factor = 1.0
        
        # Hệ số phòng ngủ
        if bedrooms >= 2:
            bedroom_factor = 0.8 + (bedrooms - 2) * 0.15
        else:
            bedroom_factor = 0.8
        
        # Hệ số phòng tắm
        if bathrooms >= 1:
            bathroom_factor = 0.8 + (bathrooms - 1) * 0.15
        else:
            bathroom_factor = 0.8
        
        # Hệ số tầng
        if floors >= 1:
            floor_factor = 0.8 + (floors - 1) * 0.1
        else:
            floor_factor = 0.8
        
        # Hệ số mặt tiền
        frontage_factor = 1.15 if frontage else 1.0
        
        # Tổng hợp
        total_factor = location_factor * area_factor * bedroom_factor * bathroom_factor * floor_factor * frontage_factor
        
        # Tính giá cơ bản
        base_price = prediction / total_factor if total_factor > 0 else prediction
        
        return {
            "base_price": round(base_price, 2),
            "location_factor": round(location_factor, 2),
            "area_factor": round(area_factor, 2),
            "bedroom_factor": round(bedroom_factor, 2),
            "bathroom_factor": round(bathroom_factor, 2),
            "floor_factor": round(floor_factor, 2),
            "frontage_factor": round(frontage_factor, 2),
            "total_factor": round(total_factor, 2),
            "explanation": cls._generate_explanation(
                location, area_m2, bedrooms, bathrooms, floors, frontage,
                location_factor, area_factor, bedroom_factor, bathroom_factor, 
                floor_factor, frontage_factor, prediction
            )
        }
    
    @classmethod
    def _generate_explanation(cls, location, area_m2, bedrooms, bathrooms, floors, frontage,
                             location_factor, area_factor, bedroom_factor, bathroom_factor,
                             floor_factor, frontage_factor, prediction):
        """Tạo giải thích bằng ngôn ngữ tự nhiên"""
        
        explanations = []
        
        # Giải thích về location
        if location_factor > 1.2:
            explanations.append(f"📍 Vị trí {location} có giá cao hơn trung bình (hệ số {location_factor})")
        elif location_factor < 0.8:
            explanations.append(f"📍 Vị trí {location} có giá thấp hơn trung bình (hệ số {location_factor})")
        else:
            explanations.append(f"📍 Vị trí {location} có giá ở mức trung bình")
        
        # Giải thích về diện tích
        if area_m2 > 100:
            explanations.append(f"📐 Diện tích {area_m2}m² lớn, tăng giá (hệ số {area_factor})")
        elif area_m2 < 40:
            explanations.append(f"📐 Diện tích {area_m2}m² nhỏ, giảm giá (hệ số {area_factor})")
        else:
            explanations.append(f"📐 Diện tích {area_m2}m² ở mức trung bình")
        
        # Giải thích về số phòng
        if bedrooms >= 4:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình lớn (hệ số {bedroom_factor})")
        elif bedrooms <= 2:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình nhỏ (hệ số {bedroom_factor})")
        else:
            explanations.append(f"🛏️ {bedrooms} phòng ngủ - phù hợp gia đình trung bình (hệ số {bedroom_factor})")
        
        # Giải thích về số phòng tắm
        if bathrooms >= 3:
            explanations.append(f"🛁 {bathrooms} phòng tắm - tiện nghi cao (hệ số {bathroom_factor})")
        else:
            explanations.append(f"🛁 {bathrooms} phòng tắm (hệ số {bathroom_factor})")
        
        # Giải thích về số tầng
        if floors >= 5:
            explanations.append(f"🏗️ {floors} tầng - nhà cao tầng (hệ số {floor_factor})")
        else:
            explanations.append(f"🏗️ {floors} tầng (hệ số {floor_factor})")
        
        # Giải thích về mặt tiền
        if frontage:
            explanations.append("🚪 Có mặt tiền - tăng giá trị (hệ số 1.15)")
        else:
            explanations.append("🚪 Không có mặt tiền")
        
        # Tổng kết
        explanations.append(f"💰 Giá dự đoán: {prediction:,.0f} triệu VND")
        
        return " | ".join(explanations)


class HousePriceKnowledgeGraph:
    """Knowledge Graph cho dự đoán giá nhà"""
    
    def __init__(self):
        self.graph = {
            "nodes": [],
            "edges": [],
            "relationships": {}
        }
    
    def create_graph(self, location, area_m2, bedrooms, bathrooms, floors, frontage, prediction, factors):
        """Tạo Knowledge Graph từ input"""
        
        # Reset graph
        self.graph = {
            "nodes": [],
            "edges": [],
            "relationships": {}
        }
        
        # Thêm node chính: House
        self._add_node("house", "🏠 Nhà", "Chính", {
            "Vị trí": location,
            "Diện tích": f"{area_m2}m²",
            "Phòng ngủ": f"{bedrooms} phòng",
            "Phòng tắm": f"{bathrooms} phòng",
            "Số tầng": f"{floors} tầng",
            "Mặt tiền": "Có" if frontage else "Không",
            "Giá dự đoán": f"{prediction:,.0f} triệu VND"
        })
        
        # Thêm các node yếu tố ảnh hưởng
        self._add_node("location", "📍 Vị trí", "Yếu tố", {
            "Tên": location,
            "Hệ số": factors.get("location_factor", 1.0),
            "Tác động": "Cao" if factors.get("location_factor", 1.0) > 1.2 else "Thấp" if factors.get("location_factor", 1.0) < 0.8 else "Trung bình"
        })
        
        self._add_node("area", "📐 Diện tích", "Yếu tố", {
            "Giá trị": f"{area_m2}m²",
            "Hệ số": factors.get("area_factor", 1.0),
            "Tác động": "Lớn" if area_m2 > 100 else "Nhỏ" if area_m2 < 40 else "Trung bình"
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
        
        # Thêm node giá
        self._add_node("price", "💰 Giá", "Kết quả", {
            "Giá trị": f"{prediction:,.0f} triệu VND"
        })
        
        # Thêm các kết nối (edges)
        self._add_edge("location", "house", "ảnh hưởng", factors.get("location_factor", 1.0))
        self._add_edge("area", "house", "ảnh hưởng", factors.get("area_factor", 1.0))
        self._add_edge("bedrooms", "house", "ảnh hưởng", factors.get("bedroom_factor", 1.0))
        self._add_edge("bathrooms", "house", "ảnh hưởng", factors.get("bathroom_factor", 1.0))
        self._add_edge("floors", "house", "ảnh hưởng", factors.get("floor_factor", 1.0))
        self._add_edge("frontage", "house", "ảnh hưởng", factors.get("frontage_factor", 1.0))
        self._add_edge("house", "price", "dẫn đến", 1.0)
        
        # Thêm relationships
        self.graph["relationships"] = {
            "location": {
                "type": "location",
                "description": f"Vị trí {location} ảnh hưởng {factors.get('location_factor', 1.0)}x đến giá"
            },
            "area": {
                "type": "area",
                "description": f"Diện tích {area_m2}m² ảnh hưởng {factors.get('area_factor', 1.0)}x đến giá"
            },
            "bedrooms": {
                "type": "bedrooms",
                "description": f"{bedrooms} phòng ngủ ảnh hưởng {factors.get('bedroom_factor', 1.0)}x đến giá"
            },
            "bathrooms": {
                "type": "bathrooms",
                "description": f"{bathrooms} phòng tắm ảnh hưởng {factors.get('bathroom_factor', 1.0)}x đến giá"
            },
            "floors": {
                "type": "floors",
                "description": f"{floors} tầng ảnh hưởng {factors.get('floor_factor', 1.0)}x đến giá"
            },
            "frontage": {
                "type": "frontage",
                "description": f"Mặt tiền ảnh hưởng {factors.get('frontage_factor', 1.0)}x đến giá"
            }
        }
        
        return self.graph
    
    def _add_node(self, node_id, label, node_type, properties):
        """Thêm node vào graph"""
        self.graph["nodes"].append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "properties": properties
        })
    
    def _add_edge(self, source, target, label, weight):
        """Thêm edge vào graph"""
        self.graph["edges"].append({
            "source": source,
            "target": target,
            "label": label,
            "weight": weight
        })
    
    def get_graph(self):
        """Lấy toàn bộ graph"""
        return self.graph