# layout_analyzer.py
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

class LayoutAnalyzer:
    """
    Analyzes document layout to identify structural elements such as
    paragraphs, headings, tables, images, and other visual components.
    """
    
    def __init__(self, min_block_size: int = 50):
        """
        Initialize the layout analyzer.
        
        Args:
            min_block_size: Minimum size of a text block to be considered significant
        """
        self.min_block_size = min_block_size
    
    def analyze(self, page_image: np.ndarray, page_text: str, page_idx: int) -> List[Dict[str, Any]]:
        """
        Analyze the layout of a document page.
        
        Args:
            page_image: Image representation of the page
            page_text: Extracted text from the page
            page_idx: Index of the page in the document
            
        Returns:
            List of detected elements with their bounding boxes and type
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect text blocks
        text_blocks = self._detect_text_blocks(binary)
        
        # Detect tables
        tables = self._detect_tables(binary, text_blocks)
        
        # Detect images
        images = self._detect_images(page_image, binary)
        
        # Create elements list
        elements = []
        
        # Add text blocks
        for idx, (x, y, w, h) in enumerate(text_blocks):
            # Skip blocks that overlap with tables
            if any(self._rectangles_overlap((x, y, w, h), table) for table in tables):
                continue
                
            elements.append({
                "type": "text_block",
                "id": f"page_{page_idx}_text_{idx}",
                "bbox": (x, y, x + w, y + h),
                "content": self._extract_text_in_region(page_text, (x, y, w, h))
            })
        
        # Add tables
        for idx, (x, y, w, h) in enumerate(tables):
            elements.append({
                "type": "table",
                "id": f"page_{page_idx}_table_{idx}",
                "bbox": (x, y, x + w, y + h)
            })
        
        # Add images
        for idx, (x, y, w, h) in enumerate(images):
            elements.append({
                "type": "image",
                "id": f"page_{page_idx}_image_{idx}",
                "bbox": (x, y, x + w, y + h)
            })
        
        return elements
    
    def _detect_text_blocks(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text blocks in the binary image."""
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size
        text_blocks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > self.min_block_size:
                text_blocks.append((x, y, w, h))
        
        return text_blocks
    
    def _detect_tables(self, binary_image: np.ndarray, 
                      text_blocks: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Detect tables using line detection and text block analysis."""
        # Create a copy of the binary image
        table_binary = binary_image.copy()
        
        # Apply morphological operations to connect table lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(table_binary, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, horizontal_kernel)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and merge table contours
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the region contains text blocks
            text_block_count = sum(
                1 for tx, ty, tw, th in text_blocks 
                if self._rectangle_inside((tx, ty, tw, th), (x, y, w, h))
            )
            
            if text_block_count >= 4 and w > 100 and h > 100:
                tables.append((x, y, w, h))
        
        # Merge overlapping tables
        return self._merge_overlapping_rectangles(tables)
    
    def _detect_images(self, page_image: np.ndarray, 
                      binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect images in the page."""
        # Create a mask for potential image regions
        image_mask = np.zeros_like(binary_image)
        
        # Convert to HSV and extract saturation channel
        hsv = cv2.cvtColor(page_image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Threshold saturation to find colorful regions (likely images)
        _, sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours in saturation mask
        contours, _ = cv2.findContours(
            sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter image contours
        images = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip small regions
            if w < 50 or h < 50:
                continue
                
            # Check average saturation in the region
            roi = saturation[y:y+h, x:x+w]
            avg_saturation = np.mean(roi)
            
            # Images typically have higher saturation
            if avg_saturation > 30:
                images.append((x, y, w, h))
        
        return images
    
    def _extract_text_in_region(self, page_text: str, region: Tuple[int, int, int, int]) -> str:
        """
        Extract text that falls within a region.
        This is a simplified implementation and would need to be refined in a real system.
        """
        # In a real implementation, this would use the coordinates of each character
        # Since pdfplumber doesn't easily provide character coordinates, this is a placeholder
        return ""
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                           rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _rectangle_inside(self, inner_rect: Tuple[int, int, int, int], 
                         outer_rect: Tuple[int, int, int, int]) -> bool:
        """Check if inner rectangle is inside outer rectangle."""
        x1, y1, w1, h1 = inner_rect
        x2, y2, w2, h2 = outer_rect
        
        return (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)
    
    def _merge_overlapping_rectangles(self, 
                                     rectangles: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping rectangles."""
        if not rectangles:
            return []
            
        # Sort rectangles by x coordinate
        sorted_rects = sorted(rectangles, key=lambda r: r[0])
        
        merged = [sorted_rects[0]]
        for rect in sorted_rects[1:]:
            last = merged[-1]
            
            # If current rectangle overlaps with last merged rectangle, merge them
            if self._rectangles_overlap(last, rect):
                # Create a new rectangle that encompasses both
                x = min(last[0], rect[0])
                y = min(last[1], rect[1])
                w = max(last[0] + last[2], rect[0] + rect[2]) - x
                h = max(last[1] + last[3], rect[1] + rect[3]) - y
                
                # Replace the last rectangle with merged one
                merged[-1] = (x, y, w, h)
            else:
                merged.append(rect)
        
        return merged
