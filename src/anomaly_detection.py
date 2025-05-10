"""
Anomaly detection module for blood analysis.
Contains functions for detecting and classifying blood cell anomalies.
"""

import numpy as np
from typing import List, Dict, Any
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path: Optional path to a pretrained model
        """
        self.model = None
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)

    def extract_features(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Extract features from a cell image for anomaly detection.
        
        Args:
            cell_image: Image of a single cell
            
        Returns:
            np.ndarray: Feature vector
        """
        # Resize to standard size
        resized = tf.image.resize(cell_image, (64, 64))
        
        # Extract basic features
        features = []
        
        # Add shape features
        features.append(cell_image.shape[0] / cell_image.shape[1])  # aspect ratio
        
        # Add intensity features
        features.extend([
            np.mean(cell_image),
            np.std(cell_image),
            np.median(cell_image),
        ])
        
        return np.array(features)

    def detect_anomalies(self, cells: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a list of cell images.
        
        Args:
            cells: List of cell images
            
        Returns:
            List of dictionaries containing anomaly information for each cell
        """
        results = []
        
        for cell in cells:
            features = self.extract_features(cell)
            
            # For now, use a simple threshold-based approach
            # This should be replaced with a proper ML model
            mean_intensity = np.mean(cell)
            std_intensity = np.std(cell)
            
            anomaly = {
                'is_anomaly': std_intensity > 50,  # Example threshold
                'confidence': float(std_intensity / 100),
                'features': {
                    'mean_intensity': float(mean_intensity),
                    'std_intensity': float(std_intensity),
                }
            }
            
            results.append(anomaly)
        
        return results

    def load_model(self, model_path: str):
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, model_path: str):
        """
        Save the current model.
        
        Args:
            model_path: Path where to save the model
        """
        if self.model is not None:
            self.model.save(model_path)
