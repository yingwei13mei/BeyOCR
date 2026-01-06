# large_document_processor.py
import numpy as np
from typing import List, Dict, Any, Tuple

class LargeDocumentProcessor:
    """
    Processes large documents using a sliding window approach to
    manage memory constraints while maintaining context across windows.
    """
    
    def __init__(self, base_processor, window_size=5, overlap=1):
        """
        Initialize with a base document processor and window parameters.
        
        Args:
            base_processor: The document processor to use for each window
            window_size: Number of pages to process in each window
            overlap: Number of overlapping pages between windows
        """
        self.base_processor = base_processor
        self.window_size = window_size
        self.overlap = overlap
    
    def process_large_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a large document using sliding windows.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Combined document graph
        """
        import pdfplumber
        
        # Get total number of pages
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        # Calculate window parameters
        stride = self.window_size - self.overlap
        num_windows = max(1, 1 + (total_pages - self.window_size) // stride)
        
        # Process each window
        all_window_results = []
        for i in range(num_windows):
            start_page = i * stride
            end_page = min(start_page + self.window_size, total_pages)
            
            print(f"Processing window {i+1}/{num_windows}: pages {start_page+1} to {end_page}")
            
            # Extract window using temporary file
            window_pdf_path = self._extract_pdf_window(pdf_path, start_page, end_page)
            
            # Process window
            window_result = self.base_processor.process_document(window_pdf_path)
            
            # Add window metadata
            self._add_window_metadata(window_result, start_page, end_page, total_pages)
            
            all_window_results.append(window_result)
        
        # Merge window results
        merged_result = self._merge_window_results(all_window_results)
        
        return merged_result
    
    def _extract_pdf_window(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """
        Extract a window of pages from a PDF to a temporary file.
        
        In a real implementation, this would use a PDF library to extract pages.
        Here we use a placeholder implementation.
        """
        import tempfile
        from PyPDF2 import PdfReader, PdfWriter
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Extract pages to the temporary file
        pdf_reader = PdfReader(pdf_path)
        pdf_writer = PdfWriter()
        
        for page_num in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])
            
        with open(temp_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        return temp_path
    
    def _add_window_metadata(self, window_result: Dict[str, Any], 
                            start_page: int, end_page: int, total_pages: int):
        """Add window metadata to result for later merging."""
        window_result["_window_metadata"] = {
            "start_page": start_page,
            "end_page": end_page,
            "total_pages": total_pages
        }
        
        # Adjust page numbers in nodes
        for node in window_result.get("nodes", []):
            if "page" in node:
                node["original_page"] = node["page"]
                node["page"] += start_page
                
            if "id" in node and "_page_" in node["id"]:
                parts = node["id"].split("_page_")
                if len(parts) == 2:
                    prefix, rest = parts
                    page_num, element_info = rest.split("_", 1)
                    try:
                        adjusted_page = int(page_num) + start_page
                        node["id"] = f"{prefix}_page_{adjusted_page}_{element_info}"
                    except ValueError:
                        pass
    
    def _merge_window_results(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from all windows into a unified document graph.
        
        Args:
            window_results: List of window processing results
            
        Returns:
            Merged document graph
        """
        if not window_results:
            return {}
            
        # Collect all nodes and edges
        all_nodes = []
        all_edges = []
        
        node_id_mapping = {}  # To handle potential duplicate IDs
        
        for window_idx, window in enumerate(window_results):
            # Process nodes
            for node in window.get("nodes", []):
                # Ensure unique ID
                original_id = node.get("id")
                if original_id in node_id_mapping:
                    new_id = f"{original_id}_window_{window_idx}"
                    node_id_mapping[original_id] = new_id
                    node["id"] = new_id
                else:
                    node_id_mapping[original_id] = original_id
                
                all_nodes.append(node)
            
            # Process edges
            for edge in window.get("edges", []):
                # Update source and target with mapped IDs
                source = edge.get("source")
                target = edge.get("target")
                
                if source in node_id_mapping:
                    edge["source"] = node_id_mapping[source]
                
                if target in node_id_mapping:
                    edge["target"] = node_id_mapping[target]
                
                all_edges.append(edge)
        
        # Create cross-window relationships
        cross_window_edges = self._create_cross_window_relationships(
            window_results, node_id_mapping
        )
        all_edges.extend(cross_window_edges)
        
        # Merge into final result
        merged_result = {
            "nodes": all_nodes,
            "edges": all_edges
        }
        
        return merged_result
    
    def _create_cross_window_relationships(self, window_results: List[Dict[str, Any]], 
                                         node_id_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Create relationships between elements across different windows.
        
        This is crucial for maintaining document coherence across window boundaries.
        """
        cross_window_edges = []
        
        # Connect last page of each window to first page of next window
        for i in range(len(window_results) - 1):
            current_window = window_results[i]
            next_window = window_results[i+1]
            
            current_metadata = current_window.get("_window_metadata", {})
            next_metadata = next_window.get("_window_metadata", {})
            
            # Find elements on last page of current window
            last_page = current_metadata.get("end_page", 0) - 1
            last_page_elements = [
                node for node in current_window.get("nodes", [])
                if node.get("page") == last_page
            ]
            
            # Find elements on first page of next window
            first_page = next_metadata.get("start_page", 0)
            first_page_elements = [
                node for node in next_window.get("nodes", [])
                if node.get("page") == first_page
            ]
            
            # Connect last element to first element
            if last_page_elements and first_page_elements:
                # Sort by vertical position
                last_page_elements.sort(
                    key=lambda n: n.get("bbox", (0, 0, 0, 0))[3]  # Bottom y-coordinate
                )
                first_page_elements.sort(
                    key=lambda n: n.get("bbox", (0, 0, 0, 0))[1]  # Top y-coordinate
                )
                
                # Get last and first elements
                last_element = last_page_elements[-1]
                first_element = first_page_elements[0]
                
                # Create cross-window edge
                last_id = node_id_mapping.get(last_element.get("id", ""))
                first_id = node_id_mapping.get(first_element.get("id", ""))
                
                if last_id and first_id:
                    cross_window_edges.append({
                        "source": last_id,
                        "target": first_id,
                        "relationship": "follows",
                        "cross_window": True
                    })
        
        return cross_window_edges
