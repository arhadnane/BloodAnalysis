"""
Image processing module for blood sample analysis.
Contains functions for loading, preprocessing, and saving processed images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List, Dict
from skimage import filters, exposure, morphology, measure


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load and validate a microscope image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Loaded image in BGR format
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise ValueError(f"Image not found: {path}")
        
    # Read image using binary mode to handle special characters
    with open(path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
        
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image from {path}")
        
    return img


def save_processed_image(image: np.ndarray, 
                        output_path: Union[str, Path], 
                        prefix: str = "",
                        create_dirs: bool = True) -> Path:
    """
    Save a processed image to disk.
    
    Args:
        image: Image to save
        output_path: Directory or full path to save the image
        prefix: Optional prefix to add to the filename
        create_dirs: Create directories if they don't exist
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    
    # If output_path is a directory, create a filename
    if output_path.suffix == "":
        timestamp = np.datetime64('now').astype(str).replace(":", "-")
        filename = f"{prefix}_processed_{timestamp}.png" if prefix else f"processed_{timestamp}.png"
        output_path = output_path / filename
    
    # Create directories if needed
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Save the image
    cv2.imwrite(str(output_path), image)
    return output_path


def process_and_save_analysis(input_path: Union[str, Path],
                            output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Process a blood sample image and save all analysis results.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save processed images
        
    Returns:
        Dictionary with paths to all saved files
    """
    # Load image
    image = load_image(input_path)
    output_dir = Path(output_dir)
    saved_files = {}
    
    # Save original
    saved_files['original'] = save_processed_image(
        image, 
        output_dir, 
        prefix="original"
    )
    
    # Preprocess and save
    preprocessed = preprocess_image(image)
    saved_files['preprocessed'] = save_processed_image(
        preprocessed,
        output_dir,
        prefix="preprocessed"
    )
    
    # Detect cells and save visualization
    binary, contours = detect_cells(preprocessed)
    
    # Save binary mask
    saved_files['binary_mask'] = save_processed_image(
        binary,
        output_dir,
        prefix="binary_mask"
    )
    
    # Create and save cell detection visualization
    detection_vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(detection_vis, contours, -1, (0, 255, 0), 2)
    saved_files['cell_detection'] = save_processed_image(
        detection_vis,
        output_dir,
        prefix="cell_detection"
    )
    
    # Save individual cell analysis
    for i, contour in enumerate(contours):
        # Get cell region
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = preprocessed[y:y+h, x:x+w]
        
        # Segment components
        components = segment_cell_components(cell_image)
        
        # Create RGB visualization of components
        cell_vis = np.zeros((h, w, 3), dtype=np.uint8)
        cell_vis[components['nucleus'] > 0] = [255, 0, 0]    # Red for nucleus
        cell_vis[components['cytoplasm'] > 0] = [0, 255, 0]  # Green for cytoplasm
        
        saved_files[f'cell_{i+1}'] = save_processed_image(
            cell_vis,
            output_dir,
            prefix=f"cell_{i+1}"
        )
        
    return saved_files


def preprocess_image(image: np.ndarray, 
                    denoise_strength: int = 10,
                    clahe_clip: float = 2.0,
                    clahe_grid: int = 8) -> np.ndarray:
    """
    Preprocess the blood sample image for analysis.
    
    Args:
        image: Input image in BGR format
        denoise_strength: Strength of denoising (h parameter)
        clahe_clip: Clip limit for CLAHE
        clahe_grid: Grid size for CLAHE
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gray)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        denoise_strength,
        21  # templateWindowSize
    )
    
    return denoised


def detect_cells(image: np.ndarray, 
                min_area: int = 50,
                max_area: int = 10000,
                circularity_thresh: float = 0.3) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect and segment blood cells in the image.
    
    Args:
        image: Preprocessed grayscale image
        min_area: Minimum cell area in pixels (reduced for small cells)
        max_area: Maximum cell area in pixels (increased for large cells)
        circularity_thresh: Minimum circularity (0-1) for cell detection (relaxed for real samples)
        
    Returns:
        Tuple containing:
        - Binary mask of detected cells
        - List of contours for each detected cell
    """
    # Make sure we have a grayscale image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize image to full range
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
      # Use adaptive thresholding for better local contrast
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=151,  # Large block size for blood cells
        C=10  # Increased contrast threshold
    )
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours based on area and shape
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:  # Skip invalid contours
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:  # Skip invalid contours
            continue
            
        # Calculate shape metrics
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Apply filters
        if (min_area <= area <= max_area and 
            circularity >= circularity_thresh and
            solidity >= 0.8):  # Must be reasonably solid
            valid_contours.append(cnt)
    
    return binary, valid_contours
    
    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours based on area and circularity with more relaxed threshold
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
            
            # More comprehensive filtering
            if (min_area < area < max_area and 
                circularity > circularity_thresh and
                solidity > 0.85):  # Additional check for cell-like shapes
                valid_contours.append(cnt)
    
    return binary, valid_contours


def analyze_cell_properties(image: np.ndarray, 
                          contour: np.ndarray) -> Dict[str, float]:
    """
    Extract various properties from a detected cell.
    
    Args:
        image: Preprocessed grayscale image
        contour: Contour of the cell
        
    Returns:
        Dictionary containing cell properties
    """
    # Create mask for the cell
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Get cell region
    x, y, w, h = cv2.boundingRect(contour)
    cell_region = image[y:y+h, x:x+w]
    mask_region = mask[y:y+h, x:x+w]
    
    # Calculate properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Intensity properties
    cell_pixels = cell_region[mask_region == 255]
    mean_intensity = np.mean(cell_pixels)
    std_intensity = np.std(cell_pixels)
    
    # Calculate basic texture features
    if len(cell_pixels) > 0:
        # Calculate local contrast
        contrast = np.std(cell_pixels) / (np.max(cell_pixels) - np.min(cell_pixels) + 1e-6)
        
        # Calculate homogeneity using local variation
        dx = cv2.Sobel(cell_region, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(cell_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(dx**2 + dy**2)
        homogeneity = 1.0 / (1.0 + np.mean(gradient_mag[mask_region == 255]))
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'contrast': float(contrast),
        'homogeneity': float(homogeneity)
    }


def segment_cell_components(cell_image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segment different components within a cell (nucleus, cytoplasm).
    
    Args:
        cell_image: Image of a single cell
        
    Returns:
        Dictionary containing masks for different cell components
    """
    # Convert to float and rescale to 0-1
    img_float = cell_image.astype(float) / 255.0
    
    # Multi-level Otsu thresholding
    thresh_multi = filters.threshold_multiotsu(img_float, classes=3)
    regions = np.digitize(img_float, bins=thresh_multi)
    
    # Create masks for different components
    nucleus_mask = (regions >= 2)
    cytoplasm_mask = (regions == 1)
    background_mask = (regions == 0)
    
    # Convert to uint8 and scale to 0-255
    nucleus_mask = nucleus_mask.astype(np.uint8) * 255
    cytoplasm_mask = cytoplasm_mask.astype(np.uint8) * 255
    background_mask = background_mask.astype(np.uint8) * 255
    
    # Clean up masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for mask in [nucleus_mask, cytoplasm_mask, background_mask]:
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)
    
    return {
        'nucleus': nucleus_mask,
        'cytoplasm': cytoplasm_mask,
        'background': background_mask
    }


def calculate_population_statistics(cell_measurements: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical measures for cell population characteristics.
    
    Args:
        cell_measurements: List of dictionaries containing cell measurements
        
    Returns:
        Dictionary of statistics for each measured property
    """
    if not cell_measurements:
        return {}
        
    # Initialize stats dictionary
    stats = {}
    
    # Get all measurement keys from first cell
    properties = cell_measurements[0].keys()
    
    # Calculate statistics for each property
    for prop in properties:
        values = [cell[prop] for cell in cell_measurements]
        stats[prop] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
        
        # Add quartiles
        q1, q3 = np.percentile(values, [25, 75])
        stats[prop]['q1'] = float(q1)
        stats[prop]['q3'] = float(q3)
        stats[prop]['iqr'] = float(q3 - q1)
        
        # Identify outliers using 1.5 * IQR rule
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        stats[prop]['outlier_count'] = len(outliers)
        stats[prop]['outlier_fraction'] = len(outliers) / len(values)
        
    return stats
