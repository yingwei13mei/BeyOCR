# element_classifier.py
import numpy as np
from typing import Dict, List, Any
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class ElementClassifier:
    """
    Classifies document elements based on their appearance and content.
    Uses vision models to determine semantic roles (e.g., heading, paragraph, footer).
    """
    
    def __init__(self, model_name="microsoft/layoutlm-document-classification"):
        """
        Initialize with a document element classification model.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        # In a production system, we'd load an actual model here
        # For this example, we'll use a simulated classifier
        self.feature_extractor = None
        self.model = None
        
    def classify(self, elements: List[Dict[str, Any]], page_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify each element in the given list.
        
        Args:
            elements: List of document elements to classify
            page_image: Image of the page for visual classification
            
        Returns:
            Updated list of elements with classification information
        """
        classified_elements = []
        
        for element in elements:
            element_type = element.get("type", "text_block")
            element_copy = element.copy()
            
            if element_type == "text_block":
                # Extract region from the page image
                if "bbox" in element:
                    x1, y1, x2, y2 = element["bbox"]
                    if (x2 > x1 and y2 > y1 and 
                        x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                        region = page_image[y1:y2, x1:x2]
                        element_copy["semantic_type"] = self._classify_text_block(
                            region, element.get("content", "")
                        )
            
            elif element_type == "table":
                element_copy["semantic_type"] = "data_table"
            
            elif element_type == "image":
                if "bbox" in element:
                    x1, y1, x2, y2 = element["bbox"]
                    if (x2 > x1 and y2 > y1 and 
                        x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                        region = page_image[y1:y2, x1:x2]
                        element_copy["semantic_type"] = self._classify_image(region)
            
            classified_elements.append(element_copy)
        
        return classified_elements
    
    def _classify_text_block(self, region: np.ndarray, text: str) -> str:
        """
        Classify a text block based on its appearance and content.
        
        In a production system, this would use an actual ML model.
        Here we use simple heuristics for demonstration.
        """
        if not text:
            return "unknown"
            
        # Simple heuristic classification based on text properties
        text_lower = text.lower()
        
        # Check for headings
        if len(text) < 100 and text.isupper():
            return "heading"
            
        # Check for footer/header
        if "page" in text_lower and len(text) < 50:
            return "page_number"
            
        # Check for captions
        if text_lower.startswith(("figure", "fig.", "table", "chart")):
            return "caption"
        
        # Default to paragraph
        return "paragraph"
    
    def _classify_image(self, region: np.ndarray) -> str:
        """
        Classify an image region (chart, diagram, photo, etc.)
        
        In a production system, this would use an image classification model.
        Here we use simple heuristics for demonstration.
        """
        # Convert to grayscale
        if region.ndim == 3:
            gray = np.mean(region, axis=2).astype(np.uint8)
        else:
            gray = region
            
        # Simple heuristics for image classification
        # Charts often have more edges/contours
        edge_density = self._calculate_edge_density(gray)
        
        if edge_density > 0.2:
            return "chart"
        else:
            return "photo"
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate the density of edges in an image."""
        import cv2
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray_image.size
        
        return edge_pixels / total_pixels if total_pixels > 0 else 0
