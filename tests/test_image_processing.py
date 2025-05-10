"""
Test cases for image processing module.
"""

import cv2
import numpy as np
import pytest
from src.image_processing import (
    load_image,
    preprocess_image,
    detect_cells,
    analyze_cell_properties,
    segment_cell_components
)


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a 200x200 image with some circle-like objects
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some circular cells
    cv2.circle(img, (50, 50), 20, 255, -1)  # Filled circle
    cv2.circle(img, (150, 150), 25, 255, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def test_load_image_invalid_path():
    """Test that loading an invalid image path raises an error"""
    with pytest.raises(ValueError):
        load_image("nonexistent_image.jpg")


def test_preprocess_image(sample_image):
    """Test image preprocessing function"""
    # Test with default parameters
    result = preprocess_image(sample_image)
    assert result.shape == sample_image.shape
    assert result.dtype == np.uint8
    
    # Test with custom parameters
    result_custom = preprocess_image(
        sample_image,
        denoise_strength=15,
        clahe_clip=3.0,
        clahe_grid=16
    )
    assert result_custom.shape == sample_image.shape
    
    # Result should be different with different parameters
    assert not np.array_equal(result, result_custom)


def test_detect_cells(sample_image):
    """Test cell detection function"""
    binary, contours = detect_cells(sample_image)
    
    # Check binary mask
    assert binary.shape == sample_image.shape
    assert binary.dtype == np.uint8
    assert np.unique(binary).tolist() == [0, 255]  # Binary image
    
    # Should detect 2 cells in sample image
    assert len(contours) == 2
    
    # Test with custom parameters
    binary_custom, contours_custom = detect_cells(
        sample_image,
        min_area=50,
        max_area=10000,
        circularity_thresh=0.5
    )
    
    # Should still detect the circles
    assert len(contours_custom) >= 2


def test_analyze_cell_properties(sample_image):
    """Test cell property analysis"""
    # Get a cell contour
    _, contours = detect_cells(sample_image)
    assert len(contours) > 0
    
    # Analyze first cell
    props = analyze_cell_properties(sample_image, contours[0])
    
    # Check required properties
    required_props = [
        'area', 'perimeter', 'circularity',
        'mean_intensity', 'std_intensity',
        'contrast', 'homogeneity'
    ]
    
    for prop in required_props:
        assert prop in props
        assert isinstance(props[prop], float)
    
    # Test property values
    assert props['area'] > 0
    assert props['perimeter'] > 0
    assert 0 <= props['circularity'] <= 1
    assert 0 <= props['homogeneity'] <= 1


def test_segment_cell_components(sample_image):
    """Test cell component segmentation"""
    # Get a cell region
    _, contours = detect_cells(sample_image)
    assert len(contours) > 0
    
    # Get first cell region
    x, y, w, h = cv2.boundingRect(contours[0])
    cell_image = sample_image[y:y+h, x:x+w]
    
    # Segment components
    components = segment_cell_components(cell_image)
    
    # Check required components
    required_components = ['nucleus', 'cytoplasm', 'background']
    
    for comp in required_components:
        assert comp in components
        assert components[comp].shape == cell_image.shape
        assert components[comp].dtype == np.uint8
        assert np.unique(components[comp]).tolist() == [0, 255]  # Binary masks


def test_preprocess_image_color_input():
    """Test preprocessing with color input image"""
    # Create color test image
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(color_image, (50, 50), 20, (255, 255, 255), -1)
    
    result = preprocess_image(color_image)
    assert len(result.shape) == 2  # Should convert to grayscale
    assert result.dtype == np.uint8


def test_cell_detection_edge_cases(sample_image):
    """Test cell detection with edge cases"""
    # Test with empty image
    empty_image = np.zeros_like(sample_image)
    binary, contours = detect_cells(empty_image)
    assert len(contours) == 0
    
    # Test with very high threshold
    binary, contours = detect_cells(
        sample_image,
        min_area=1000000  # Unreasonably large
    )
    assert len(contours) == 0
    
    # Test with very low circularity threshold
    binary, contours = detect_cells(
        sample_image,
        circularity_thresh=0.1
    )
    assert len(contours) >= 2  # Should detect at least our circles
