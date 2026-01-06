# relationship_mapper.py
import networkx as nx
from typing import Dict, List, Any

class RelationshipMapper:
    """
    Maps relationships between document elements to create a knowledge graph
    representation of the document.
    """
    
    def map_relationships(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a graph representation of element relationships.
        
        Args:
            elements: List of document elements
            
        Returns:
            Document graph with nodes and relationships
        """
        # Create a graph
        graph = nx.DiGraph()
        
        # Add all elements as nodes
        for element in elements:
            element_id = element.get("id", f"element_{len(graph.nodes)}")
            graph.add_node(element_id, **element)
        
        # Map hierarchical relationships (e.g., heading -> subsequent paragraphs)
        self._map_hierarchical_relationships(graph, elements)
        
        # Map sequential relationships (reading order)
        self._map_sequential_relationships(graph, elements)
        
        # Map reference relationships (e.g., text referring to figures)
        self._map_reference_relationships(graph, elements)
        
        # Convert graph to dictionary representation
        document_graph = {
            "nodes": [self._node_to_dict(node, data) for node, data in graph.nodes(data=True)],
            "edges": [self._edge_to_dict(u, v, data) for u, v, data in graph.edges(data=True)]
        }
        
        return document_graph
    
    def _map_hierarchical_relationships(self, graph: nx.DiGraph, elements: List[Dict[str, Any]]):
        """Map hierarchical relationships between elements."""
        # Group elements by page
        pages = {}
        for element in elements:
            page = element.get("page", 0)
            if page not in pages:
                pages[page] = []
            pages[page].append(element)
        
        # Process each page
        for page, page_elements in pages.items():
            # Sort elements by y-coordinate
            sorted_elements = sorted(page_elements, key=lambda e: e.get("bbox", (0, 0, 0, 0))[1])
            
            # Find headings
            headings = [e for e in sorted_elements if e.get("semantic_type") == "heading"]
            
            # For each heading, find elements that follow it
            for i, heading in enumerate(headings):
                heading_id = heading.get("id")
                heading_y = heading.get("bbox", (0, 0, 0, 0))[3]  # Bottom y-coordinate
                
                # Find the y-coordinate of the next heading
                next_heading_y = float('inf')
                if i < len(headings) - 1:
                    next_heading_y = headings[i+1].get("bbox", (0, 0, 0, 0))[1]  # Top y-coordinate
                
                # Find elements between this heading and the next
                for element in sorted_elements:
                    element_id = element.get("id")
                    if element_id == heading_id:
                        continue
                        
                    element_y = element.get("bbox", (0, 0, 0, 0))[1]  # Top y-coordinate
                    
                    # If element is between current heading and next heading
                    if heading_y <= element_y < next_heading_y:
                        graph.add_edge(heading_id, element_id, relationship="contains")
    
    def _map_sequential_relationships(self, graph: nx.DiGraph, elements: List[Dict[str, Any]]):
        """Map sequential relationships (reading order) between elements."""
        # Group elements by page
        pages = {}
        for element in elements:
            page = element.get("page", 0)
            if page not in pages:
                pages[page] = []
            pages[page].append(element)
        
        # Connect sequential elements within each page
        for page, page_elements in pages.items():
            # Sort elements by position (top-to-bottom, left-to-right)
            sorted_elements = sorted(
                page_elements, 
                key=lambda e: (e.get("bbox", (0, 0, 0, 0))[1], e.get("bbox", (0, 0, 0, 0))[0])
            )
            
            # Connect elements in reading order
            for i in range(len(sorted_elements) - 1):
                current_id = sorted_elements[i].get("id")
                next_id = sorted_elements[i+1].get("id")
                graph.add_edge(current_id, next_id, relationship="follows")
        
        # Connect last element of each page to first element of next page
        page_nums = sorted(pages.keys())
        for i in range(len(page_nums) - 1):
            current_page = page_nums[i]
            next_page = page_nums[i+1]
            
            if current_page in pages and next_page in pages:
                current_page_elements = sorted(
                    pages[current_page],
                    key=lambda e: (e.get("bbox", (0, 0, 0, 0))[1], e.get("bbox", (0, 0, 0, 0))[0])
                )
                next_page_elements = sorted(
                    pages[next_page],
                    key=lambda e: (e.get("bbox", (0, 0, 0, 0))[1], e.get("bbox", (0, 0, 0, 0))[0])
                )
                
                if current_page_elements and next_page_elements:
                    last_element_id = current_page_elements[-1].get("id")
                    first_element_id = next_page_elements[0].get("id")
                    graph.add_edge(last_element_id, first_element_id, relationship="follows")
    
    def _map_reference_relationships(self, graph: nx.DiGraph, elements: List[Dict[str, Any]]):
        """Map reference relationships (e.g., text referring to figures)."""
        # Find image elements
        images = [e for e in elements if e.get("type") in ["image", "table"]]
        
        # Find text elements
        text_elements = [e for e in elements if e.get("type") == "text_block"]
        
        # For each text element, check if it references any image
        for text_element in text_elements:
            text_id = text_element.get("id")
            content = text_element.get("content", "").lower()
            
            for image in images:
                image_id = image.get("id")
                image_type = image.get("type")
                
                # Simple reference detection based on element id
                # In a real system, this would use more sophisticated NLP
                if image_type == "image" and "figure" in content and image_id.split("_")[-1] in content:
                    graph.add_edge(text_id, image_id, relationship="references")
                elif image_type == "table" and "table" in content and image_id.split("_")[-1] in content:
                    graph.add_edge(text_id, image_id, relationship="references")
    
    def _node_to_dict(self, node_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a graph node to dictionary representation."""
        result = {"id": node_id}
        result.update({k: v for k, v in data.items() if k != "id"})
        return result
    
    def _edge_to_dict(self, source: str, target: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a graph edge to dictionary representation."""
        result = {"source": source, "target": target}
        result.update(data)
        return result
