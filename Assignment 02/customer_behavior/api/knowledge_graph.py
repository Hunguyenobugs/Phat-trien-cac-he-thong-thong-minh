# api/knowledge_graph.py
import json
from pathlib import Path

class ProductKnowledgeGraph:
    def __init__(self):
        # Lưu trữ dữ liệu dưới dạng dictionary
        self.products = {}
        self.relationships = {}
        self._build_graph()
    
    def _build_graph(self):
        """Xây dựng knowledge graph từ dữ liệu sản phẩm"""
        
        # Dữ liệu sản phẩm
        product_data = {
            "Dresses": {
                "category": "Dresses",
                "department": "Dresses",
                "division": "General",
                "styles": ["casual", "formal", "party", "summer"],
                "occasions": ["wedding", "party", "work", "date"],
                "materials": ["cotton", "silk", "polyester"]
            },
            "Knits": {
                "category": "Knits",
                "department": "Tops",
                "division": "General",
                "styles": ["casual", "warm", "cozy"],
                "occasions": ["casual", "work", "home"],
                "materials": ["wool", "cotton", "acrylic"]
            },
            "Blouses": {
                "category": "Blouses",
                "department": "Tops",
                "division": "General",
                "styles": ["formal", "casual", "elegant"],
                "occasions": ["work", "date", "formal"],
                "materials": ["silk", "cotton", "polyester"]
            },
            "Sweaters": {
                "category": "Sweaters",
                "department": "Tops",
                "division": "General",
                "styles": ["warm", "cozy", "casual"],
                "occasions": ["casual", "home", "winter"],
                "materials": ["wool", "cotton", "cashmere"]
            },
            "Pants": {
                "category": "Pants",
                "department": "Bottoms",
                "division": "General",
                "styles": ["casual", "formal", "comfortable"],
                "occasions": ["work", "casual", "travel"],
                "materials": ["cotton", "polyester", "denim"]
            },
            "Jeans": {
                "category": "Jeans",
                "department": "Bottoms",
                "division": "General",
                "styles": ["casual", "street", "trendy"],
                "occasions": ["casual", "weekend", "travel"],
                "materials": ["denim", "cotton"]
            },
            "Skirts": {
                "category": "Skirts",
                "department": "Bottoms",
                "division": "General",
                "styles": ["casual", "formal", "flattering"],
                "occasions": ["work", "party", "date"],
                "materials": ["cotton", "polyester", "wool"]
            },
            "Jackets": {
                "category": "Jackets",
                "department": "Jackets",
                "division": "General",
                "styles": ["warm", "casual", "stylish"],
                "occasions": ["casual", "work", "travel"],
                "materials": ["leather", "wool", "polyester"]
            },
            "Outerwear": {
                "category": "Outerwear",
                "department": "Jackets",
                "division": "General",
                "styles": ["warm", "winter", "rain"],
                "occasions": ["winter", "travel", "casual"],
                "materials": ["wool", "polyester", "nylon"]
            },
            "Intimates": {
                "category": "Intimates",
                "department": "Intimate",
                "division": "Initmates",
                "styles": ["sexy", "comfortable", "lace"],
                "occasions": ["home", "date"],
                "materials": ["lace", "silk", "cotton"]
            },
            "Lounge": {
                "category": "Lounge",
                "department": "Intimate",
                "division": "Initmates",
                "styles": ["comfortable", "relaxed", "cozy"],
                "occasions": ["home", "sleep"],
                "materials": ["cotton", "modal", "polyester"]
            },
            "Sleep": {
                "category": "Sleep",
                "department": "Intimate",
                "division": "Initmates",
                "styles": ["comfortable", "sleep"],
                "occasions": ["home", "sleep"],
                "materials": ["cotton", "silk", "modal"]
            },
            "Swim": {
                "category": "Swim",
                "department": "Intimate",
                "division": "Initmates",
                "styles": ["beach", "sexy", "sporty"],
                "occasions": ["beach", "pool", "vacation"],
                "materials": ["nylon", "polyester", "spandex"]
            },
            "Shorts": {
                "category": "Shorts",
                "department": "Bottoms",
                "division": "General",
                "styles": ["casual", "summer", "sporty"],
                "occasions": ["casual", "summer", "beach"],
                "materials": ["cotton", "linen", "polyester"]
            },
            "Fine gauge": {
                "category": "Fine gauge",
                "department": "Tops",
                "division": "General",
                "styles": ["elegant", "light", "knit"],
                "occasions": ["work", "casual"],
                "materials": ["cotton", "silk", "wool"]
            },
            "Layering": {
                "category": "Layering",
                "department": "Tops",
                "division": "General",
                "styles": ["versatile", "light", "casual"],
                "occasions": ["casual", "work", "travel"],
                "materials": ["cotton", "polyester", "modal"]
            },
            "Legwear": {
                "category": "Legwear",
                "department": "Bottoms",
                "division": "General",
                "styles": ["comfortable", "basic"],
                "occasions": ["casual", "work"],
                "materials": ["cotton", "nylon", "spandex"]
            },
            "Trend": {
                "category": "Trend",
                "department": "Trend",
                "division": "General",
                "styles": ["fashionable", "modern", "stylish"],
                "occasions": ["party", "date", "casual"],
                "materials": ["polyester", "viscose", "cotton"]
            }
        }
        
        self.products = product_data
        
        # Xây dựng các mối quan hệ
        self.relationships = {}
        
        # Thêm các mối quan hệ cho từng sản phẩm
        for product_name, attrs in product_data.items():
            self.relationships[product_name] = {
                "belongs_to_division": attrs["division"],
                "belongs_to_department": attrs["department"],
                "has_styles": attrs["styles"],
                "good_for": attrs["occasions"],
                "made_of": attrs["materials"]
            }
        
        # Xây dựng các mối quan hệ giữa các sản phẩm cùng department
        self.product_relations = {}
        departments = self._get_all_departments()
        
        for dept in departments:
            products_in_dept = [p for p, attrs in product_data.items() if attrs["department"] == dept]
            for p1 in products_in_dept:
                if p1 not in self.product_relations:
                    self.product_relations[p1] = []
                for p2 in products_in_dept:
                    if p1 != p2 and p2 not in self.product_relations[p1]:
                        self.product_relations[p1].append({
                            "product": p2,
                            "relation": "related_by_department"
                        })
        
        # Xây dựng các mối quan hệ dựa trên style
        styles = self._get_all_styles()
        for style in styles:
            products_with_style = [p for p, attrs in product_data.items() if style in attrs["styles"]]
            for p1 in products_with_style:
                if p1 not in self.product_relations:
                    self.product_relations[p1] = []
                for p2 in products_with_style:
                    if p1 != p2 and not any(r["product"] == p2 for r in self.product_relations[p1]):
                        self.product_relations[p1].append({
                            "product": p2,
                            "relation": "related_by_style"
                        })
        
        # Xây dựng các mối quan hệ dựa trên occasions
        occasions = self._get_all_occasions()
        for occasion in occasions:
            products_with_occasion = [p for p, attrs in product_data.items() if occasion in attrs["occasions"]]
            for p1 in products_with_occasion:
                if p1 not in self.product_relations:
                    self.product_relations[p1] = []
                for p2 in products_with_occasion:
                    if p1 != p2 and not any(r["product"] == p2 for r in self.product_relations[p1]):
                        self.product_relations[p1].append({
                            "product": p2,
                            "relation": "related_by_occasion"
                        })
    
    def _get_all_departments(self):
        departments = set()
        for attrs in self.products.values():
            departments.add(attrs["department"])
        return list(departments)
    
    def _get_all_styles(self):
        styles = set()
        for attrs in self.products.values():
            for style in attrs["styles"]:
                styles.add(style)
        return list(styles)
    
    def _get_all_occasions(self):
        occasions = set()
        for attrs in self.products.values():
            for occasion in attrs["occasions"]:
                occasions.add(occasion)
        return list(occasions)
    
    def _get_all_materials(self):
        materials = set()
        for attrs in self.products.values():
            for material in attrs["materials"]:
                materials.add(material)
        return list(materials)
    
    def get_product_info(self, product_name):
        """Lấy thông tin chi tiết về một sản phẩm"""
        if product_name not in self.products:
            return None
        
        product_data = dict(self.products[product_name])
        product_data["name"] = product_name
        
        # Thêm các kết nối
        connections = {}
        if product_name in self.relationships:
            for key, value in self.relationships[product_name].items():
                if key not in connections:
                    connections[key] = []
                if isinstance(value, list):
                    connections[key].extend(value)
                else:
                    connections[key].append(value)
        
        product_data["connections"] = connections
        
        # Thêm các sản phẩm liên quan
        related = []
        if product_name in self.product_relations:
            for rel in self.product_relations[product_name]:
                related.append({
                    "name": rel["product"],
                    "relation": rel["relation"],
                    "attributes": dict(self.products.get(rel["product"], {}))
                })
        product_data["related_products"] = related
        
        return product_data
    
    def get_related_products(self, product_name, relation=None):
        """Lấy các sản phẩm liên quan"""
        if product_name not in self.product_relations:
            return []
        
        related = []
        for rel in self.product_relations[product_name]:
            if relation is None or rel["relation"] == relation:
                related.append({
                    "name": rel["product"],
                    "relation": rel["relation"],
                    "attributes": dict(self.products.get(rel["product"], {}))
                })
        return related
    
    def recommend_products(self, product_name, max_results=5):
        """Gợi ý sản phẩm dựa trên sản phẩm đã chọn"""
        if product_name not in self.products:
            return []
        
        scores = {}
        product_attrs = self.products[product_name]
        
        # Duyệt qua tất cả sản phẩm khác
        for other_name, other_attrs in self.products.items():
            if other_name == product_name:
                continue
            
            score = 0
            
            # Cùng department: +3 điểm
            if other_attrs["department"] == product_attrs["department"]:
                score += 3
            
            # Cùng style: +2 điểm
            common_styles = set(other_attrs["styles"]) & set(product_attrs["styles"])
            score += len(common_styles) * 2
            
            # Cùng occasion: +1 điểm
            common_occasions = set(other_attrs["occasions"]) & set(product_attrs["occasions"])
            score += len(common_occasions) * 1
            
            # Cùng material: +1 điểm
            common_materials = set(other_attrs["materials"]) & set(product_attrs["materials"])
            score += len(common_materials) * 1
            
            if score > 0:
                scores[other_name] = score
        
        # Sắp xếp và trả về kết quả
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_products[:max_results]]
    
    def find_common_interests(self, product1, product2):
        """Tìm điểm chung giữa hai sản phẩm"""
        if product1 not in self.products or product2 not in self.products:
            return []
        
        attrs1 = self.products[product1]
        attrs2 = self.products[product2]
        
        common = []
        
        # So sánh styles
        common_styles = set(attrs1["styles"]) & set(attrs2["styles"])
        for style in common_styles:
            common.append({
                "name": style,
                "type": "style"
            })
        
        # So sánh occasions
        common_occasions = set(attrs1["occasions"]) & set(attrs2["occasions"])
        for occasion in common_occasions:
            common.append({
                "name": occasion,
                "type": "occasion"
            })
        
        # So sánh materials
        common_materials = set(attrs1["materials"]) & set(attrs2["materials"])
        for material in common_materials:
            common.append({
                "name": material,
                "type": "material"
            })
        
        return common
    
    def get_all_products(self):
        """Lấy danh sách tất cả sản phẩm"""
        return sorted(self.products.keys())
    
    def get_knowledge_graph_data(self, product_name):
        """Lấy dữ liệu knowledge graph cho một sản phẩm (dạng nodes và edges)"""
        if product_name not in self.products:
            return None
        
        nodes = []
        edges = []
        
        # Thêm node sản phẩm chính
        nodes.append({
            "id": product_name,
            "label": product_name,
            "group": "product"
        })
        
        attrs = self.products[product_name]
        
        # Thêm nodes và edges cho các thuộc tính
        # Division
        division = attrs["division"]
        nodes.append({
            "id": division,
            "label": division,
            "group": "division"
        })
        edges.append({
            "from": product_name,
            "to": division,
            "label": "belongs_to_division"
        })
        
        # Department
        department = attrs["department"]
        nodes.append({
            "id": department,
            "label": department,
            "group": "department"
        })
        edges.append({
            "from": product_name,
            "to": department,
            "label": "belongs_to_department"
        })
        
        # Styles
        for style in attrs["styles"]:
            nodes.append({
                "id": style,
                "label": style,
                "group": "style"
            })
            edges.append({
                "from": product_name,
                "to": style,
                "label": "has_style"
            })
        
        # Occasions
        for occasion in attrs["occasions"]:
            nodes.append({
                "id": occasion,
                "label": occasion,
                "group": "occasion"
            })
            edges.append({
                "from": product_name,
                "to": occasion,
                "label": "good_for"
            })
        
        # Materials
        for material in attrs["materials"]:
            nodes.append({
                "id": material,
                "label": material,
                "group": "material"
            })
            edges.append({
                "from": product_name,
                "to": material,
                "label": "made_of"
            })
        
        # Loại bỏ các node trùng lặp
        unique_nodes = {}
        for node in nodes:
            if node["id"] not in unique_nodes:
                unique_nodes[node["id"]] = node
        nodes = list(unique_nodes.values())
        
        return {
            "nodes": nodes,
            "edges": edges
        }