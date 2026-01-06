# document_extraction_evaluator.py
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import precision_recall_fscore_support

class DocumentExtractionEvaluator:
    """
    Evaluates document extraction system performance against ground truth.
    """
    
    def evaluate(self, extracted_graph: Dict[str, Any], 
                ground_truth_graph: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate extraction performance against ground truth.
        
        Args:
            extracted_graph: Document graph from extraction system
            ground_truth_graph: Ground truth document graph
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Evaluate element detection
        element_metrics = self._evaluate_element_detection(
            extracted_graph.get("nodes", []),
            ground_truth_graph.get("nodes", [])
        )
        results.update(element_metrics)
        
        # Evaluate element classification
        classification_metrics = self._evaluate_element_classification(
            extracted_graph.get("nodes", []),
            ground_truth_graph.get("nodes", [])
        )
        results.update(classification_metrics)
        
        # Evaluate relationship extraction
        relationship_metrics = self._evaluate_relationships(
            extracted_graph.get("edges", []),
            ground_truth_graph.get("edges", [])
        )
        results.update(relationship_metrics)
        
        # Calculate overall score
        results["overall_score"] = np.mean([
            results.get("element_detection_f1", 0),
            results.get("classification_accuracy", 0),
            results.get("relationship_f1", 0)
        ])
        
        return results
    
    def _evaluate_element_detection(self, extracted_nodes: List[Dict[str, Any]], 
                                   ground_truth_nodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate element detection performance."""
        # Match elements based on bounding box overlap
        matched_pairs = self._match_elements_by_bbox(extracted_nodes, ground_truth_nodes)
        
        # Calculate precision and recall
        if not ground_truth_nodes:
            return {"element_detection_precision": 0, "element_detection_recall": 0, "element_detection_f1": 0}
            
        precision = len(matched_pairs) / len(extracted_nodes) if extracted_nodes else 0
        recall = len(matched_pairs) / len(ground_truth_nodes)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "element_detection_precision": precision,
            "element_detection_recall": recall,
            "element_detection_f1": f1
        }
    
    def _evaluate_element_classification(self, extracted_nodes: List[Dict[str, Any]], 
                                       ground_truth_nodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate element classification performance."""
        # Match elements based on bounding box overlap
        matched_pairs = self._match_elements_by_bbox(extracted_nodes, ground_truth_nodes)
        
        if not matched_pairs:
            return {"classification_accuracy": 0}
        
        # Check type matches
        correct_classifications = 0
        for extracted_idx, gt_idx in matched_pairs:
            extracted_type = extracted_nodes[extracted_idx].get("semantic_type", "unknown")
            gt_type = ground_truth_nodes[gt_idx].get("semantic_type", "unknown")
            
            if extracted_type == gt_type:
                correct_classifications += 1
        
        accuracy = correct_classifications / len(matched_pairs) if matched_pairs else 0
        
        return {"classification_accuracy": accuracy}
    
    def _evaluate_relationships(self, extracted_edges: List[Dict[str, Any]], 
                              ground_truth_edges: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate relationship extraction performance."""
        # Create sets of relationships for comparison
        extracted_relationships = set(
            (edge.get("source", ""), edge.get("target", ""), edge.get("relationship", ""))
            for edge in extracted_edges
        )
        
        ground_truth_relationships = set(
            (edge.get("source", ""), edge.get("target", ""), edge.get("relationship", ""))
            for edge in ground_truth_edges
        )
        
        # Calculate metrics
        true_positives = len(extracted_relationships.intersection(ground_truth_relationships))
        
        if not ground_truth_relationships:
            return {"relationship_precision": 0, "relationship_recall": 0, "relationship_f1": 0}
            
        precision = true_positives / len(extracted_relationships) if extracted_relationships else 0
        recall = true_positives / len(ground_truth_relationships)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "relationship_precision": precision,
            "relationship_recall": recall,
            "relationship_f1": f1
        }
    
    def _match_elements_by_bbox(self, extracted_nodes: List[Dict[str, Any]], 
                              ground_truth_nodes: List[Dict[str, Any]]) -> List[tuple]:
        """Match extracted elements to ground truth elements based on bbox overlap."""
        matched_pairs = []
        
        for i, extracted_node in enumerate(extracted_nodes):
            extracted_bbox = extracted_node.get("bbox")
            if not extracted_bbox:
                continue
                
            best_match = -1
            best_iou = 0.5  # Minimum IoU threshold
            
            for j, gt_node in enumerate(ground_truth_nodes):
                gt_bbox = gt_node.get("bbox")
                if not gt_bbox:
                    continue
                    
                # Calculate IoU
                iou = self._calculate_bbox_iou(extracted_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
            
            if best_match >= 0:
                matched_pairs.append((i, best_match))
        
        return matched_pairs
    
    def _calculate_bbox_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection area
        x_intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_intersection = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection_area = x_intersection * y_intersection
        
        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
