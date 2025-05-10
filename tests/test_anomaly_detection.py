"""
Test cases for anomaly detection module.
"""

import pytest
import numpy as np
from src.anomaly_detection import AnomalyDetector


def test_anomaly_detector_initialization():
    """Test AnomalyDetector initialization"""
    detector = AnomalyDetector()
    assert detector.model is None
    assert detector.scaler is not None


def test_feature_extraction():
    """Test feature extraction from cell images"""
    # Create a test cell image
    cell_image = np.random.randint(0, 255, (50, 40), dtype=np.uint8)
    
    detector = AnomalyDetector()
    features = detector.extract_features(cell_image)
    
    # Check feature properties
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1  # Should be a 1D feature vector
    assert len(features) > 0
    assert not np.any(np.isnan(features))  # No NaN values


def test_anomaly_detection():
    """Test anomaly detection on cell images"""
    # Create test cells
    cells = [
        np.random.randint(0, 255, (40, 40), dtype=np.uint8),
        np.random.randint(0, 255, (50, 45), dtype=np.uint8)
    ]
    
    detector = AnomalyDetector()
    results = detector.detect_anomalies(cells)
    
    # Check results
    assert len(results) == len(cells)
    for result in results:
        assert 'is_anomaly' in result
        assert isinstance(result['is_anomaly'], bool)
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
        assert 'features' in result
        assert isinstance(result['features'], dict)
