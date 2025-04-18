"""
Tests for deployment utilities.
"""
import pytest
import numpy as np
import cv2
import sys
import os
from pathlib import Path
import time
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.utils import get_device_info, ResourceMonitor
from src.deployment.edge_pipeline import EdgeInferenceEngine

# Skip tests that require a model file if none is available
has_test_model = os.path.exists('tests/test_data/test_model.onnx')


class TestDeviceInfo:
    """Test cases for device information utilities."""
    
    def test_get_device_info(self):
        """Test retrieving device information."""
        info = get_device_info()
        
        # Check that basic info is present
        assert 'cpu' in info
        assert 'cores' in info['cpu']
        assert 'memory' in info
        assert 'total' in info['memory']
        assert 'platform' in info
        
        # Check types
        assert isinstance(info['cpu']['cores'], int)
        assert isinstance(info['memory']['total'], int)
        assert isinstance(info['memory']['percent_used'], float)


class TestResourceMonitor:
    """Test cases for resource monitoring."""
    
    def test_resource_monitor(self):
        """Test resource monitoring functionality."""
        # Create a temporary directory for logs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create monitor
            monitor = ResourceMonitor(log_dir=temp_dir, interval=0.1)
            
            # Start monitoring
            monitor.start()
            
            # Wait a short time
            time.sleep(0.5)
            
            # Stop monitoring
            stats = monitor.stop()
            
            # Check that statistics were collected
            assert len(stats['timestamps']) > 0
            assert len(stats['cpu']) == len(stats['timestamps'])
            assert len(stats['memory']) == len(stats['timestamps'])
            
            # Check that a log file was created
            log_files = list(Path(temp_dir).glob('*.json'))
            assert len(log_files) > 0


@pytest.mark.skipif(not has_test_model, reason="Test model not available")
class TestEdgeInferenceEngine:
    """Test cases for edge inference engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Path to test model
        self.model_path = 'tests/test_data/test_model.onnx'
        
        # Input size for model
        self.input_size = (224, 224)
        
        # Create engine
        self.engine = EdgeInferenceEngine(
            model_path=self.model_path,
            input_size=self.input_size,
            use_gpu=False
        )
        
    def test_preprocess(self):
        """Test image preprocessing."""
        # Create test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), -1)
        
        # Preprocess image
        blob = self.engine.preprocess(image)
        
        # Check shape
        assert blob.shape[0] == 1  # Batch size
        assert blob.shape[1] == 3  # Channels
        assert blob.shape[2:] == self.input_size  # Height, width
