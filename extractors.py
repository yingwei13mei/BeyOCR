# extractors.py
import numpy as np
import cv2
from typing import Dict, List, Any

class TextExtractor:
    """
    Extracts and processes text blocks, handling different text roles
    like paragraphs, headings, and lists.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a text element to extract its content with proper structure.
        
        Args:
            element: The text element to process
            page_image: Image of the page containing the element
            
        Returns:
            Processed element with structured content
        """
        semantic_type = element.get("semantic_type", "paragraph")
        content = element.get("content", "")
        
        result = element.copy()
        
        if semantic_type == "heading":
            result["level"] = self._estimate_heading_level(content, element.get("bbox"))
        elif semantic_type == "paragraph":
            result["sentences"] = self._split_into_sentences(content)
        elif semantic_type == "list":
            result["items"] = self._extract_list_items(content)
        
        return result
    
    def _estimate_heading_level(self, text: str, bbox=None) -> int:
        """Estimate the heading level based on text properties and size."""
        # In a real implementation, font size would be a major factor
        # Here we use simple heuristics
        if text.isupper():
            return 1
        elif len(text) < 30:
            return 2
        else:
            return 3
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing."""
        import re
        return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract items from a list."""
        import re
        # Try to identify list markers and split accordingly
        items = re.split(r'\n\s*[\•\-\*\d+\.]\s+', text)
        return [item.strip() for item in items if item.strip()]

class TableExtractor:
    """
    Extracts structured data from tables in documents.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract structured data from a table element.
        
        Args:
            element: The table element to process
            page_image: Image of the page containing the table
            
        Returns:
            Processed table with structured data
        """
        result = element.copy()
        
        if "bbox" in element:
            x1, y1, x2, y2 = element["bbox"]
            if (x2 > x1 and y2 > y1 and 
                x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                table_image = page_image[y1:y2, x1:x2]
                
                # Process the table image to extract structured data
                table_data = self._process_table_image(table_image)
                result["data"] = table_data
        
        return result
    
    def _process_table_image(self, table_image: np.ndarray) -> List[List[str]]:
        """
        Process a table image to extract its structure and content.
        
        In a production system, this would use specialized table extraction models.
        Here we use a simplified approach for demonstration.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours to identify cells
        contours, _ = cv2.findContours(
            table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by position to reconstruct table structure
        # This is a simplified approach; real implementation would be more complex
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # In a real implementation, we would now:
        # 1. Identify grid structure
        # 2. Extract text from each cell using OCR
        # 3. Build a structured table
        
        # For this example, we return a mock table
        return [
            ["Header 1", "Header 2", "Header 3"],
            ["Value 1,1", "Value 1,2", "Value 1,3"],
            ["Value 2,1", "Value 2,2", "Value 2,3"]
        ]

class ImageExtractor:
    """
    Processes image elements, including charts, diagrams, and photos.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image element to extract visual information.
        
        Args:
            element: The image element to process
            page_image: Image of the page containing the element
            
        Returns:
            Processed image element with extracted information
        """
        result = element.copy()
        
        semantic_type = element.get("semantic_type", "photo")
        
        if "bbox" in element:
            x1, y1, x2, y2 = element["bbox"]
            if (x2 > x1 and y2 > y1 and 
                x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                image_region = page_image[y1:y2, x1:x2]
                
                if semantic_type == "chart":
                    chart_data = self._process_chart(image_region)
                    result["chart_data"] = chart_data
                elif semantic_type == "photo":
                    # For photos, we might extract features or descriptive tags
                    result["description"] = self._generate_image_description(image_region)
        
        return result
    
    def _process_chart(self, chart_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a chart image to extract its data.
        
        In a production system, this would use specialized chart parsing models.
        Here we use a simplified approach for demonstration.
        """
        # This is where chart parsing would occur
        # For demonstration, we return mock chart data
        return {
            "type": "bar_chart",
            "axes": {
                "x": "Categories",
                "y": "Values"
            },
            "data": [
                {"category": "A", "value": 10},
                {"category": "B", "value": 20},
                {"category": "C", "value": 15}
            ]
        }
    
    def _generate_image_description(self, image: np.ndarray) -> str:
        """
        Generate a description of an image.
        
        In a production system, this would use image captioning models.
        Here we return a placeholder description.
        """
        # For demonstration purposes
        return "Image content description would be generated here"class TextExtractor:
    """
    Extracts and processes text blocks, handling different text roles
    like paragraphs, headings, and lists.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a text element to extract its content with proper structure.
        
        Args:
            element: The text element to process
            page_image: Image of the page containing the element
            
        Returns:
            Processed element with structured content
        """
        semantic_type = element.get("semantic_type", "paragraph")
        content = element.get("content", "")
        
        result = element.copy()
        
        if semantic_type == "heading":
            result["level"] = self._estimate_heading_level(content, element.get("bbox"))
        elif semantic_type == "paragraph":
            result["sentences"] = self._split_into_sentences(content)
        elif semantic_type == "list":
            result["items"] = self._extract_list_items(content)
        
        return result
    
    def _estimate_heading_level(self, text: str, bbox=None) -> int:
        """Estimate the heading level based on text properties and size."""
        # In a real implementation, font size would be a major factor
        # Here we use simple heuristics
        if text.isupper():
            return 1
        elif len(text) < 30:
            return 2
        else:
            return 3
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better processing."""
        import re
        return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract items from a list."""
        import re
        # Try to identify list markers and split accordingly
        items = re.split(r'\n\s*[\•\-\*\d+\.]\s+', text)
        return [item.strip() for item in items if item.strip()]

class TableExtractor:
    """
    Extracts structured data from tables in documents.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract structured data from a table element.
        
        Args:
            element: The table element to process
            page_image: Image of the page containing the table
            
        Returns:
            Processed table with structured data
        """
        result = element.copy()
        
        if "bbox" in element:
            x1, y1, x2, y2 = element["bbox"]
            if (x2 > x1 and y2 > y1 and 
                x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                table_image = page_image[y1:y2, x1:x2]
                
                # Process the table image to extract structured data
                table_data = self._process_table_image(table_image)
                result["data"] = table_data
        
        return result
    
    def _process_table_image(self, table_image: np.ndarray) -> List[List[str]]:
        """
        Process a table image to extract its structure and content.
        
        In a production system, this would use specialized table extraction models.
        Here we use a simplified approach for demonstration.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours to identify cells
        contours, _ = cv2.findContours(
            table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours by position to reconstruct table structure
        # This is a simplified approach; real implementation would be more complex
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # In a real implementation, we would now:
        # 1. Identify grid structure
        # 2. Extract text from each cell using OCR
        # 3. Build a structured table
        
        # For this example, we return a mock table
        return [
            ["Header 1", "Header 2", "Header 3"],
            ["Value 1,1", "Value 1,2", "Value 1,3"],
            ["Value 2,1", "Value 2,2", "Value 2,3"]
        ]

class ImageExtractor:
    """
    Processes image elements, including charts, diagrams, and photos.
    """
    
    def extract(self, element: Dict[str, Any], page_image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image element to extract visual information.
        
        Args:
            element: The image element to process
            page_image: Image of the page containing the element
            
        Returns:
            Processed image element with extracted information
        """
        result = element.copy()
        
        semantic_type = element.get("semantic_type", "photo")
        
        if "bbox" in element:
            x1, y1, x2, y2 = element["bbox"]
            if (x2 > x1 and y2 > y1 and 
                x2 < page_image.shape[1] and y2 < page_image.shape[0]):
                image_region = page_image[y1:y2, x1:x2]
                
                if semantic_type == "chart":
                    chart_data = self._process_chart(image_region)
                    result["chart_data"] = chart_data
                elif semantic_type == "photo":
                    # For photos, we might extract features or descriptive tags
                    result["description"] = self._generate_image_description(image_region)
        
        return result
    
    def _process_chart(self, chart_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a chart image to extract its data.
        
        In a production system, this would use specialized chart parsing models.
        Here we use a simplified approach for demonstration.
        """
        # This is where chart parsing would occur
        # For demonstration, we return mock chart data
        return {
            "type": "bar_chart",
            "axes": {
                "x": "Categories",
                "y": "Values"
            },
            "data": [
                {"category": "A", "value": 10},
                {"category": "B", "value": 20},
                {"category": "C", "value": 15}
            ]
        }
    
    def _generate_image_description(self, image: np.ndarray) -> str:
        """
        Generate a description of an image.
        
        In a production system, this would use image captioning models.
        Here we return a placeholder description.
        """
        # For demonstration purposes
        return "Image content description would be generated here"
