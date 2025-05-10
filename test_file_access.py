from pathlib import Path
import os

def test_file_access():
    # Test different path approaches
    print("Current working directory:", Path.cwd())
    print("\nTrying different path approaches:")
    
    # Approach 1: Using raw string
    path1 = Path(r"data/raw/Capture d'écran 2025-01-25 225434.png")
    print(f"\n1. Raw string path: {path1}")
    print(f"   Exists: {path1.exists()}")
    print(f"   Absolute: {path1.absolute()}")
    print(f"   Resolved: {path1.resolve()}")
    
    # Approach 2: Using os.path.join
    path2 = Path(os.path.join("data", "raw", "Capture d'écran 2025-01-25 225434.png"))
    print(f"\n2. os.path.join path: {path2}")
    print(f"   Exists: {path2.exists()}")
    print(f"   Absolute: {path2.absolute()}")
    print(f"   Resolved: {path2.resolve()}")
    
    # Approach 3: List directory contents
    raw_dir = Path("data/raw")
    print(f"\n3. Directory contents of {raw_dir}:")
    try:
        for item in raw_dir.iterdir():
            print(f"   {item.name}")
    except Exception as e:
        print(f"   Error listing directory: {e}")
    
    # Approach 4: Try direct file open
    try:
        with open(path1, 'rb') as f:
            print(f"\n4. Successfully opened file with binary mode")
            data = f.read(100)  # Read first 100 bytes
            print(f"   First few bytes: {data[:10]}")
    except Exception as e:
        print(f"\n4. Error opening file: {e}")

if __name__ == "__main__":
    test_file_access()
