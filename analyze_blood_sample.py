import cv2
import numpy as np
from pathlib import Path
from src.image_processing import preprocess_image, detect_cells, analyze_cell_properties, segment_cell_components

def save_image(image: np.ndarray, output_path: Path, prefix: str = "") -> Path:
    """Save image handling special characters in path"""
    if output_path.suffix == "":
        timestamp = np.datetime64('now').astype(str).replace(":", "-")
        filename = f"{prefix}_processed_{timestamp}.png" if prefix else f"processed_{timestamp}.png"
        output_path = output_path / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        # If imwrite fails, try encoding the image first
        is_success, im_buf_arr = cv2.imencode(".png", image)
        if is_success:
            im_buf_arr.tofile(str(output_path))
    return output_path

def process_blood_sample(input_path: Path, output_dir: Path) -> None:
    """
    Process a blood sample image and save the analysis results.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save processed images
    """
    # Read image with special character handling
    img_array = np.fromfile(input_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {input_path}")
    
    print(f'Image loaded successfully. Shape: {image.shape}')
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
    for i, contour in enumerate(contours):
        print(f'\nProcessing Cell {i+1}:')
        
        # Analyze properties
        props = analyze_cell_properties(preprocessed, contour)
        
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

def main():
    try:
        # Set up paths
        input_path = Path.cwd() / 'data' / 'raw' / 'Capture d\'écran 2025-01-25 225434.png'
        output_dir = Path.cwd() / 'data' / 'processed'
        print(f'Reading from: {input_path.absolute()}')
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found at {input_path}")
            
        print(f'Output directory: {output_dir.absolute()}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the image
        process_blood_sample(input_path, output_dir)
        
        # Read image with special character handling
    img_array = np.fromfile(input_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {input_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('\nProcessing blood sample image...')
    saved_files = {}
    
    # Save original
    saved_files['original'] = save_image(image, output_dir, "original")
    
    # Preprocess
    preprocessed = preprocess_image(image)
    saved_files['preprocessed'] = save_image(preprocessed, output_dir, "preprocessed")
    
    # Detect cells
    binary, contours = detect_cells(preprocessed)
    saved_files['binary'] = save_image(binary, output_dir, "binary")
    
    # Create cell detection visualization
    detection_vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(detection_vis, contours, -1, (0, 255, 0), 2)
    saved_files['detection'] = save_image(detection_vis, output_dir, "detection")
    
    print(f'\nFound {len(contours)} cells')
    
    # Process individual cells
    cell_data = []
    for i, contour in enumerate(contours):
        # Analyze properties
        props = analyze_cell_properties(preprocessed, contour)
        cell_data.append(props)
        
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
        print(f'\nCell {i+1}:')
        print(f'  Area: {props["area"]:.2f} pixels²')
        print(f'  Circularity: {props["circularity"]:.2f}')
        print(f'  Mean Intensity: {props["mean_intensity"]:.2f}')
        print(f'  Contrast: {props["contrast"]:.2f}')
    
    print('\nProcessed files saved to:')
    for name, path in saved_files.items():
        print(f'- {name}: {path.name}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        print("\nAnalysis complete.")
