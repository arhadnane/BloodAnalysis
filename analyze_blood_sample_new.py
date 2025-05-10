"""
Script to analyze blood sample images.
"""

import cv2
import numpy as np
from pathlib import Path
from src.image_processing import (
    preprocess_image, detect_cells, analyze_cell_properties,
    segment_cell_components, calculate_population_statistics
)

def save_image(image: np.ndarray, output_path: Path, prefix: str = "") -> Path:
    """Save image handling special characters in path"""
    print(f"Saving {prefix} to {output_path}")
    try:
        if output_path.suffix == "":
            timestamp = np.datetime64('now').astype(str).replace(":", "-")
            filename = f"{prefix}_processed_{timestamp}.png" if prefix else f"processed_{timestamp}.png"
            output_path = output_path / filename
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to save with imencode/imdecode for better unicode handling
        is_success, im_buf_arr = cv2.imencode(".png", image)
        if is_success:
            im_buf_arr.tofile(str(output_path))
            return output_path
        else:
            raise ValueError("Failed to encode image")
            
    except Exception as e:
        print(f"Error saving image {prefix}: {e}")
        return None

def process_blood_sample(input_path: Path, output_dir: Path) -> None:
    """
    Process a blood sample image and save the analysis results.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save processed images
    """
    print(f"\nAttempting to read file: {input_path}")
    print(f"File exists check: {input_path.exists()}")
    print(f"Is file check: {input_path.is_file()}")
    print(f"Resolved path: {input_path.resolve()}")
    
    # Read image with special character handling
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
            print(f"Successfully read {len(data)} bytes")
            
        # Convert to numpy array and decode
        img_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        print(f'Successfully loaded image: {image.shape}')
    except Exception as e:
        print(f"Error loading image: {e}")
        raise
    
    print('\nProcessing blood sample image...')
    saved_files = {}
    
    # Save original
    saved_files['original'] = save_image(image, output_dir, "original")
    
    # Preprocess
    print('Preprocessing image...')
    preprocessed = preprocess_image(image)
    saved_files['preprocessed'] = save_image(preprocessed, output_dir, "preprocessed")
    
    # Detect cells
    print('Detecting cells...')
    binary, contours = detect_cells(preprocessed)
    saved_files['binary'] = save_image(binary, output_dir, "binary")
    
    # Create cell detection visualization
    detection_vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(detection_vis, contours, -1, (0, 255, 0), 2)
    saved_files['detection'] = save_image(detection_vis, output_dir, "detection")
    
    print(f'\nFound {len(contours)} cells')
    
    # Process individual cells
    cell_measurements = []
    for i, contour in enumerate(contours):
        print(f'\nProcessing Cell {i+1}:')
        
        # Analyze properties
        props = analyze_cell_properties(preprocessed, contour)
        cell_measurements.append(props)
        
        # Get cell region
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = preprocessed[y:y+h, x:x+w]
        
        # Segment components
        components = segment_cell_components(cell_image)
        
        # Visualize cell components
        cell_vis = np.zeros((h, w, 3), dtype=np.uint8)
        cell_vis[components['nucleus'] > 0] = [255, 0, 0]    # Red for nucleus
        cell_vis[components['cytoplasm'] > 0] = [0, 255, 0]  # Green for cytoplasm
        saved_files[f'cell_{i+1}'] = save_image(cell_vis, output_dir, f"cell_{i+1}")
        
        # Print cell analysis
        print(f'  Area: {props["area"]:.2f} pixels²')
        print(f'  Circularity: {props["circularity"]:.2f}')
        print(f'  Mean Intensity: {props["mean_intensity"]:.2f}')
        print(f'  Contrast: {props["contrast"]:.2f}')
    
    print('\nProcessed files saved to:')
    for name, path in saved_files.items():
        if path:  # Only print successfully saved files
            print(f'- {name}: {path.name}')
            
    # Calculate and display population statistics
    print("\nCalculating population statistics...")
    stats = calculate_population_statistics(cell_measurements)
    
    print("\nPopulation Statistics:")
    for prop, measures in stats.items():
        print(f"\n{prop.title()}:")
        print(f"  Mean ± Std: {measures['mean']:.2f} ± {measures['std']:.2f}")
        print(f"  Median (Q1-Q3): {measures['median']:.2f} ({measures['q1']:.2f}-{measures['q3']:.2f})")
        print(f"  Range: {measures['min']:.2f} - {measures['max']:.2f}")
        print(f"  Outliers: {measures['outlier_count']} ({measures['outlier_fraction']*100:.1f}%)")

def main():
    try:
        # Set up paths using Windows-style paths
        input_path = Path.cwd() / 'data' / 'raw' / '1.png'
        input_path = input_path.resolve()  # Get the canonical path
        output_dir = Path.cwd() / 'data' / 'processed'
        output_dir = output_dir.resolve()
        
        print(f'Reading from: {input_path}')
        print(f'Input path exists: {input_path.exists()}')
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found at {input_path}")
            
        print(f'Output directory: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the image
        process_blood_sample(input_path, output_dir)
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
