# document_graph.py
from typing import Dict, List, Any, Optional
import json
import networkx as nx

class DocumentGraph:
    """
    Represents a document as a knowledge graph for querying and analysis.
    """
    
    def __init__(self, graph_data: Dict[str, Any]):
        """
        Initialize with graph data from the RelationshipMapper.
        
        Args:
            graph_data: Dictionary with nodes and edges
        """
        self.graph_data = graph_data
        self.graph = self._build_graph(graph_data)
    
    def _build_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """Build a NetworkX graph from the graph data."""
        graph = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get("nodes", []):
            node_id = node.get("id")
            if node_id:
                graph.add_node(node_id, **node)
        
        # Add edges
        for edge in graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                edge_data = {k: v for k, v in edge.items() if k not in ["source", "target"]}
                graph.add_edge(source, target, **edge_data)
        
        return graph
    
    def query_by_type(self, element_type: str) -> List[Dict[str, Any]]:
        """
        Query elements by their type.
        
        Args:
            element_type: Type of elements to query
            
        Returns:
            List of matching elements
        """
        return [
            data for _, data in self.graph.nodes(data=True)
            if data.get("type") == element_type
        ]
    
    def query_by_semantic_type(self, semantic_type: str) -> List[Dict[str, Any]]:
        """
        Query elements by their semantic type.
        
        Args:
            semantic_type: Semantic type of elements to query
            
        Returns:
            List of matching elements
        """
        return [
            data for _, data in self.graph.nodes(data=True)
            if data.get("semantic_type") == semantic_type
        ]
    
    def get_element_context(self, element_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get an element and its contextual elements.
        
        Args:
            element_id: ID of the element
            depth: Relationship depth to include
            
        Returns:
            Element and its context
        """
        if element_id not in self.graph:
            return {}
        
        # Get the element
        element = self.graph.nodes[element_id]
        
        # Get predecessors and successors up to specified depth
        predecessors = set()
        successors = set()
        
        current_predecessors = {element_id}
        current_successors = {element_id}
        
        for _ in range(depth):
            new_predecessors = set()
            for node in current_predecessors:
                new_predecessors.update(self.graph.predecessors(node))
            predecessors.update(new_predecessors)
            current_predecessors = new_predecessors
            
            new_successors = set()
            for node in current_successors:
                new_successors.update(self.graph.successors(node))
            successors.update(new_successors)
            current_successors = new_successors
        
        # Create context
        context = {
            "element": element,
            "predecessors": [self.graph.nodes[p] for p in predecessors if p in self.graph],
            "successors": [self.graph.nodes[s] for s in successors if s in self.graph]
        }
        
        return context
    
    def get_document_hierarchy(self) -> Dict[str, Any]:
        """
        Get the hierarchical structure of the document.
        
        Returns:
            Nested dictionary representing document hierarchy
        """
        # Find root elements (typically high-level headings)
        roots = [
            node for node, in_degree in self.graph.in_degree() 
            if in_degree == 0 or (
                in_degree == 1 and 
                next(iter(self.graph.predecessors(node)), None) and
                self.graph.edges[next(iter(self.graph.predecessors(node))), node].get("relationship") == "follows"
            )
        ]
        
        # Sort roots by page and position
        def get_position(node_id):
            node = self.graph.nodes[node_id]
            page = node.get("page", 0)
            bbox = node.get("bbox", (0, 0, 0, 0))
            return (page, bbox[1])  # Sort by page, then y-coordinate
            
        roots = sorted(roots, key=get_position)
        
        # Build hierarchy
        hierarchy = []
        for root in roots:
            hierarchy.append(self._build_subtree(root))
        
        return {"document_structure": hierarchy}
    
    def _build_subtree(self, node_id: str) -> Dict[str, Any]:
        """Build a subtree of the document hierarchy."""
        node = self.graph.nodes[node_id]
        
        # Get children (contained elements)
        children = [
            child for child in self.graph.successors(node_id)
            if self.graph.edges[node_id, child].get("relationship") == "contains"
        ]
        
        # Sort children by position
        def get_position(node_id):
            node = self.graph.nodes[node_id]
            bbox = node.get("bbox", (0, 0, 0, 0))
            return bbox[1]  # Sort by y-coordinate
            
        children = sorted(children, key=get_position)
        
        # Build subtree
        subtree = {
            "id": node_id,
            "type": node.get("type", "unknown"),
            "semantic_type": node.get("semantic_type", "unknown")
        }
        
        if "content" in node:
            subtree["content"] = node["content"]
            
        if children:
            subtree["children"] = [self._build_subtree(child) for child in children]
        
        return subtree
    
    def to_json(self) -> str:
        """
        Convert the document graph to JSON string.
        
        Returns:
            JSON representation of the document graph
        """
        return json.dumps(self.graph_data, indent=2)
    
    def save(self, filepath: str):
        """
        Save the document graph to a file.
        
        Args:
            filepath: Path to save the graph
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())
