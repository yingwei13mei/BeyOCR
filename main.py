# main.py
import os
import argparse
from document_processor import DocumentProcessor
from layout_analyzer import LayoutAnalyzer
from element_classifier import ElementClassifier
from extractors import TextExtractor, TableExtractor, ImageExtractor
from relationship_mapper import RelationshipMapper
from document_graph import DocumentGraph

def main():
    """Main execution function for document extraction."""
    parser = argparse.ArgumentParser(description='Agentic Document Extraction')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('--output', '-o', default='output.json', help='Output file path')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found.")
        return
    
    print(f"Processing document: {args.pdf_path}")
    
    # Initialize components
    layout_analyzer = LayoutAnalyzer()
    element_classifier = ElementClassifier()
    
    extractors = {
        "text_block": TextExtractor(),
        "table": TableExtractor(),
        "image": ImageExtractor()
    }
    
    relationship_mapper = RelationshipMapper()
    
    # Create document processor
    processor = DocumentProcessor(
        layout_analyzer=layout_analyzer,
        element_classifier=element_classifier,
        extractors=extractors,
        relationship_mapper=relationship_mapper
    )
    
    # Process the document
    print("Extracting document information...")
    document_graph_data = processor.process_document(args.pdf_path)
    
    # Create document graph
    document_graph = DocumentGraph(document_graph_data)
    
    # Save results
    print(f"Saving results to {args.output}")
    document_graph.save(args.output)
    
    # Print summary
    print("\nDocument Processing Summary:")
    print(f"- Total elements: {len(document_graph_data.get('nodes', []))}")
    
    element_types = {}
    for node in document_graph_data.get('nodes', []):
        element_type = node.get('type', 'unknown')
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    print("- Element types:")
    for element_type, count in element_types.items():
        print(f"  - {element_type}: {count}")
    
    print("\nProcessing complete! Use the document graph for further analysis.")
if __name__ == "__main__":
    main()
