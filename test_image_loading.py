"""
Script to analyze blood sample images with robust file handling.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def find_image_file():
    """Find the blood sample image file in the raw directory."""
    # Use glob with a more permissive pattern
    pattern = os.path.join('data', 'raw', '*.png')
    files = glob.glob(pattern)
    
    if not files:
        return None
        
    # Return the largest PNG file (likely our screenshot)
    return max((Path(f) for f in files), key=lambda p: p.stat().st_size)

def main():
    try:
        print("Starting image analysis...")
        
        # Find and verify input file
        input_path = find_image_file()
        if not input_path:
            raise FileNotFoundError("Could not find PNG files in data/raw directory")
        
        print(f'Found image file: {input_path}')
        print(f'File size: {input_path.stat().st_size:,} bytes')
        
        # Read image using binary mode to handle special characters
        with open(input_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Read image using numpy and cv2
        img_array = np.fromfile(str(input_path), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Failed to decode image from {input_path}")
            
        print(f"Successfully loaded image: {image.shape}")
        
        # Save a test output to verify write permissions
        test_output = output_dir / "test_output.png"
        success = cv2.imwrite(str(test_output), image)
        print(f"Test write {'successful' if success else 'failed'}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    finally:
        print("\nProgram complete.")

if __name__ == '__main__':
    main()
