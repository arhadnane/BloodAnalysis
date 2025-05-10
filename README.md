# Blood Analysis Project

An automated system for analyzing blood sample images from microscopes to detect anomalies using computer vision and machine learning techniques.

## Features

- Load and process microscope images of blood samples
- Detect and segment individual blood cells
- Extract features from blood cells
- Detect anomalies using machine learning
- Visualize and save analysis results

## Installation

1. Clone the repository:
```powershell
git clone <repository-url>
cd BloodAnalysis
```

2. Create and activate a virtual environment (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Project Structure

```
BloodAnalysis/
├── data/               # Data directory
│   ├── raw/           # Raw microscope images
│   └── processed/     # Processed images and results
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
│   ├── __init__.py
│   ├── image_processing.py    # Image processing functions
│   ├── anomaly_detection.py   # Anomaly detection model
│   └── main.py               # Main script
├── tests/            # Unit tests
└── requirements.txt  # Project dependencies
```

## Usage

### Command Line Interface

Analyze a blood sample image:

```powershell
python src/main.py path/to/image.jpg --output results_folder --model path/to/model
```

### Python API

```python
from src.main import analyze_sample

# Analyze a single image
results = analyze_sample(
    "path/to/image.jpg",
    output_dir="results_folder",
    model_path="path/to/model"
)
```

## Development

### Running Tests

```powershell
pytest tests/
```

### Code Style

This project uses:
- Black for code formatting
- Flake8 for linting

Format code:
```powershell
black src/ tests/
```

Run linter:
```powershell
flake8 src/ tests/
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
