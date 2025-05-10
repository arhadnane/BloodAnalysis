"""
Main script for running blood sample analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from src.image_processing import load_image, preprocess_image, detect_cells
from src.anomaly_detection import AnomalyDetector


def analyze_sample(image_path: str, output_dir: str = None, model_path: str = None):
    """
    Analyze a blood sample image for anomalies.
    
    Args:
        image_path: Path to the input image
        output_dir: Optional directory for saving results
        model_path: Optional path to a pretrained model
    """
    # Load and preprocess image
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    
    # Detect cells
    mask, contours = detect_cells(preprocessed)
    
    # Extract individual cells
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell = preprocessed[y:y+h, x:x+w]
        cells.append(cell)
    
    # Detect anomalies
    detector = AnomalyDetector(model_path)
    results = detector.detect_anomalies(cells)
    
    # Save results if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save annotated image
        annotated = image.copy()
        for contour, result in zip(contours, results):
            color = (0, 0, 255) if result['is_anomaly'] else (0, 255, 0)
            cv2.drawContours(annotated, [contour], -1, color, 2)
        
        cv2.imwrite(str(output_dir / 'annotated.png'), annotated)
        
        # Save results summary
        with open(output_dir / 'results.txt', 'w') as f:
            f.write(f"Total cells detected: {len(cells)}\n")
            f.write(f"Anomalies found: {sum(1 for r in results if r['is_anomaly'])}\n")
            
            for i, result in enumerate(results):
                if result['is_anomaly']:
                    f.write(f"\nAnomaly {i+1}:\n")
                    f.write(f"Confidence: {result['confidence']:.2f}\n")
                    f.write("Features:\n")
                    for k, v in result['features'].items():
                        f.write(f"  {k}: {v:.2f}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Blood sample analysis tool")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--model", "-m", help="Path to pretrained model")
    
    args = parser.parse_args()
    analyze_sample(args.image_path, args.output, args.model)
