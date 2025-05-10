from pathlib import Path
from src.image_processing import process_and_save_analysis

def main():    # Process the image
    input_path = Path.cwd() / 'data' / 'raw' / 'Capture d\'écran 2025-01-25 225434.png'
    output_dir = Path.cwd() / 'data' / 'processed'
    print(f'Reading from: {input_path.absolute()}')

    print('Processing blood sample image...')
    saved_files = process_and_save_analysis(input_path, output_dir)

    print('\nProcessed files saved:')
    for name, path in saved_files.items():
        print(f'- {name}: {path}')

if __name__ == '__main__':
    main()
