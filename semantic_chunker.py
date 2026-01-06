# semantic_chunker.py
from typing import List, Dict, Any

class SemanticChunker:
    """
    Creates semantically meaningful chunks from document graph for RAG systems.
    """
    
    def __init__(self, max_chunk_size=1000, overlap=100):
        """
        Initialize the semantic chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def create_chunks(self, document_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantically meaningful chunks from a document graph.
        
        Args:
            document_graph: Document graph from processor
            
        Returns:
            List of chunks with metadata
        """
        # Extract document hierarchy
        if "document_structure" in document_graph:
            hierarchy = document_graph.get("document_structure", [])
        else:
            # Build flat hierarchy from nodes
            hierarchy = self._build_flat_hierarchy(document_graph.get("nodes", []))
        
        # Create chunks based on document structure
        chunks = self._chunk_hierarchy(hierarchy)
        
        return chunks
    
    def _build_flat_hierarchy(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a flat hierarchy from nodes if structured hierarchy is not available."""
        # Sort nodes by page and position
        sorted_nodes = sorted(
            nodes, 
            key=lambda n: (n.get("page", 0), n.get("bbox", (0, 0, 0, 0))[1])
        )
        
        # Create flat hierarchy
        hierarchy = []
        for node in sorted_nodes:
            if "content" in node:
                hierarchy.append({
                    "id": node.get("id", ""),
                    "type": node.get("type", "unknown"),
                    "semantic_type": node.get("semantic_type", "unknown"),
                    "content": node.get("content", "")
                })
        
        return hierarchy
    
    def _chunk_hierarchy(self, hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create chunks from document hierarchy respecting semantic boundaries.
        
        Args:
            hierarchy: Document hierarchy
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        current_chunk = {
            "text": "",
            "metadata": {
                "elements": [],
                "page_range": [float('inf'), 0]
            }
        }
        
        # Process each element in hierarchy
        self._process_hierarchy_elements(hierarchy, current_chunk, chunks)
        
        # Add last chunk if not empty
        if current_chunk["text"] and current_chunk["text"].strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _process_hierarchy_elements(self, elements: List[Dict[str, Any]], 
                                   current_chunk: Dict[str, Any], 
                                   chunks: List[Dict[str, Any]]):
        """Process hierarchy elements recursively to create chunks."""
        for element in elements:
            element_type = element.get("type", "unknown")
            semantic_type = element.get("semantic_type", "unknown")
            
            # Check if this is a section boundary
            is_section_boundary = (
                semantic_type == "heading" or 
                element.get("level", 0) in [1, 2, 3]
            )
            
            # Start new chunk at section boundaries if current chunk is not empty
            if is_section_boundary and current_chunk["text"] and len(current_chunk["text"]) > self.overlap:
                chunks.append(current_chunk)
                # Create new chunk with overlap
                overlap_text = current_chunk["text"][-self.overlap:] if self.overlap > 0 else ""
                current_chunk = {
                    "text": overlap_text,
                    "metadata": {
                        "elements": [],
                        "page_range": [float('inf'), 0]
                    }
                }
            
            # Add element content to current chunk
            if "content" in element:
                element_text = element["content"]
                current_chunk["text"] += "\n" + element_text
                
                # Update metadata
                current_chunk["metadata"]["elements"].append({
                    "id": element.get("id", ""),
                    "type": element_type,
                    "semantic_type": semantic_type
                })
                
                # Update page range
                if "page" in element:
                    page = element["page"]
                    current_page_range = current_chunk["metadata"]["page_range"]
                    current_page_range[0] = min(current_page_range[0], page)
                    current_page_range[1] = max(current_page_range[1], page)
            
            # Handle special elements (tables, charts)
            if element_type in ["table", "chart"] and "data" in element:
                # For tables, add structured representation
                table_data = element.get("data", [])
                if table_data:
                    table_text = self._format_table_as_text(table_data)
                    current_chunk["text"] += "\n" + table_text
            
            # Process children recursively
            if "children" in element:
                self._process_hierarchy_elements(
                    element["children"], current_chunk, chunks
                )
            
            # Check if current chunk exceeds max size
            if len(current_chunk["text"]) >= self.max_chunk_size:
                chunks.append(current_chunk)
                # Create new chunk with overlap
                overlap_text = current_chunk["text"][-self.overlap:] if self.overlap > 0 else ""
                current_chunk = {
                    "text": overlap_text,
                    "metadata": {
                        "elements": [],
                        "page_range": current_chunk["metadata"]["page_range"].copy()
                    }
                }
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as text for inclusion in chunks."""
        if not table_data:
            return ""
            
        table_text = "Table:\n"
        
        # Calculate column widths
        col_widths = [0] * len(table_data[0])
        for row in table_data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format header
        if table_data:
            header = table_data[0]
            header_text = " | ".join(
                str(cell).ljust(col_widths[i]) 
                for i, cell in enumerate(header) if i < len(col_widths)
            )
            table_text += header_text + "\n"
            
            # Add separator
            separator = "-" * len(header_text)
            table_text += separator + "\n"
        
        # Format data rows
        for row in table_data[1:]:
            row_text = " | ".join(
                str(cell).ljust(col_widths[i])
                for i, cell in enumerate(row) if i < len(col_widths)
            )
            table_text += row_text + "\n"
        
        return table_text
