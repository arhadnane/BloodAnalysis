"""
Script to analyze blood sample images with robust file handling.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import shutil

def find_blood_sample():
    """Find the blood sample image in raw directory."""
    raw_dir = Path('data/raw')
    png_files = list(raw_dir.glob('*.png'))
    
    if not png_files:
        return None
    
    # Return the largest PNG file (likely our screenshot)
    return max(png_files, key=lambda p: p.stat().st_size)

def save_image_safely(image, output_path):
    """Save image with special character handling."""
    try:
        # Convert path to string and normalize it
        output_path = str(Path(output_path).resolve())
        
        # Try imencode/imdecode approach first
        is_success, buffer = cv2.imencode('.png', image)
        if not is_success:
            return False
            
        with open(output_path, 'wb') as f:
            f.write(buffer)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def main():
    try:
        print("Starting blood sample analysis...")
        
        # Locate the blood sample image
        input_path = find_blood_sample()
        if not input_path:
            raise FileNotFoundError("No PNG files found in data/raw directory")
        
        print(f'Found image: {input_path.name}')
        print(f'File size: {input_path.stat().st_size:,} bytes')
        
        # Ensure the processed directory exists
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read the image file in binary mode
        with open(input_path, 'rb') as f:
            data = f.read()
            
        # Convert to numpy array and decode
        img_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        print(f'Successfully loaded image: {image.shape}')
        
        # Try to save a verification image
        verification_path = output_dir / 'verification.png'
        if save_image_safely(image, verification_path):
            print(f'Successfully saved verification image to {verification_path}')
            
            # Verify the saved image can be loaded
            test_load = cv2.imread(str(verification_path))
            if test_load is not None:
                print('Verification image loaded successfully')
            else:
                print('Warning: Could not reload verification image')
        else:
            print('Failed to save verification image')
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProgram complete.")

if __name__ == '__main__':
    main()
