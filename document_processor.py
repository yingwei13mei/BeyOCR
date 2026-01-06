# document_processor.py
import os
import pdfplumber
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional

class DocumentProcessor:
    """
    Main orchestrator for the document extraction pipeline.
    Coordinates the process of transforming a PDF into a structured representation.
    """
    
    def __init__(self, layout_analyzer=None, element_classifier=None, 
                 extractors=None, relationship_mapper=None):
        """
        Initialize the document processor with its component subsystems.
        
        Args:
            layout_analyzer: Component for identifying document structure
            element_classifier: Component for classifying document elements
            extractors: Dictionary of specialized extractors for different element types
            relationship_mapper: Component for mapping relationships between elements
        """
        self.layout_analyzer = layout_analyzer
        self.element_classifier = element_classifier
        self.extractors = extractors or {}
        self.relationship_mapper = relationship_mapper
        
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document and extract its structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            A dictionary containing the structured representation of the document
        """
        # Load and preprocess the document
        document = self._load_document(pdf_path)
        
        # Extract layout elements from each page
        all_elements = []
        for page_idx, page in enumerate(document.pages):
            # Convert page to image for visual analysis
            page_image = self._page_to_image(page)
            
            # Extract text and layout information
            page_text = page.extract_text()
            
            # Analyze layout to identify structural elements
            if self.layout_analyzer:
                page_elements = self.layout_analyzer.analyze(
                    page_image=page_image,
                    page_text=page_text,
                    page_idx=page_idx
                )
            else:
                # Fallback to simple text extraction if no layout analyzer
                page_elements = [{"type": "text", "content": page_text}]
            
            # Classify elements if classifier is available
            if self.element_classifier and page_elements:
                page_elements = self.element_classifier.classify(page_elements, page_image)
            
            # Process each element with specialized extractors
            processed_elements = []
            for element in page_elements:
                element_type = element.get("type", "text")
                extractor = self.extractors.get(element_type)
                
                if extractor:
                    processed_element = extractor.extract(element, page_image)
                else:
                    processed_element = element
                
                processed_element["page"] = page_idx
                processed_elements.append(processed_element)
            
            all_elements.extend(processed_elements)
        
        # Map relationships between elements
        if self.relationship_mapper and all_elements:
            document_graph = self.relationship_mapper.map_relationships(all_elements)
        else:
            document_graph = {"elements": all_elements}
        
        return document_graph
    
    def _load_document(self, pdf_path: str):
        """Load a PDF document using pdfplumber."""
        return pdfplumber.open(pdf_path)
    
    def _page_to_image(self, page):
        """Convert a PDF page to an image for visual analysis."""
        img = page.to_image(resolution=300)
        return np.array(img.original)
