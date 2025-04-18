"""
Tests for MobileNeRF model.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mobilenerf import (
    MobileNeRFEncoder, 
    PositionalEncoding, 
    MobileNeRFRenderer, 
    MobileNeRF
)


class TestMobileNeRFEncoder:
    """Test cases for MobileNeRFEncoder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a small encoder for testing
        self.encoder = MobileNeRFEncoder(
            in_channels=3,
            latent_dim=32,
            base_channels=16
        )
        
    def test_forward(self):
        """Test forward pass of encoder."""
        # Create dummy input
        x = torch.randn(2, 3, 64, 64)
        
        # Run forward pass
        output = self.encoder(x)
        
        # Check output shape
        # Input: [2, 3, 64, 64]
        # Expected output: [2, 32, 8, 8] (64/8 = 8 due to three encoder blocks)
        assert output.shape == (2, 32, 8, 8)
        
    def test_parameter_count(self):
        """Test that the parameter count is reasonable for edge devices."""
        param_count = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        
        # Ensure the model is not too large for edge devices
        assert param_count < 1000000  # Less than 1M parameters


class TestPositionalEncoding:
    """Test cases for PositionalEncoding."""
    
    def test_encoding_shape(self):
        """Test shape of encoded output."""
        encoding = PositionalEncoding(num_freqs=4, include_input=True)
        x = torch.randn(2, 10, 3)  # Batch of 10 3D positions
        
        encoded = encoding(x)
        
        # Expected shape: [2, 10, 3 + 3*2*4]
        # Original 3 + sin/cos for each of 3 dimensions at 4 frequencies
        assert encoded.shape == (2, 10, 3 + 3*2*4)
        
    def test_include_input_param(self):
        """Test include_input parameter."""
        # With include_input=False
        encoding = PositionalEncoding(num_freqs=4, include_input=False)
        x = torch.randn(2, 10, 3)
        
        encoded = encoding(x)
        
        # Expected shape: [2, 10, 3*2*4] (no original input)
        assert encoded.shape == (2, 10, 3*2*4)


class TestMobileNeRFRenderer:
    """Test cases for MobileNeRFRenderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = MobileNeRFRenderer(
            latent_dim=32,
            pos_encoding_freqs=4,
            hidden_dim=64,
            num_layers=2
        )
        
    def test_forward(self):
        """Test forward pass of renderer."""
        # Create dummy inputs
        batch_size = 2
        num_points = 100
        ray_points = torch.randn(batch_size, num_points, 3)
        ray_dirs = torch.randn(batch_size, num_points, 3)
        latent_features = torch.randn(batch_size, 32, 8, 8)
        
        # Run forward pass
        density, color = self.renderer(ray_points, ray_dirs, latent_features)
        
        # Check output shapes
        assert density.shape == (batch_size, num_points, 1)
        assert color.shape == (batch_size, num_points, 3)
        
        # Check value ranges
        assert torch.all(density >= 0)  # Density should be non-negative
        assert torch.all(color >= 0) and torch.all(color <= 1)  # Color should be in [0,1]


class TestMobileNeRF:
    """Test cases for MobileNeRF."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model = MobileNeRF(
            latent_dim=32,
            base_channels=16,
            num_samples=32,
            pos_encoding_freqs=4,
            hidden_dim=64,
            num_layers=2
        )
        
    def test_generate_rays(self):
        """Test ray generation."""
        batch_size = 2
        height = 32
        width = 32
        focal_length = 50
        
        # Generate rays
        origins, directions = self.model.generate_rays(
            batch_size, height, width, focal_length
        )
        
        # Check shapes
        assert origins.shape == (batch_size, height, width, 3)
        assert directions.shape == (batch_size, height, width, 3)
        
        # Check that directions are normalized
        norm = torch.norm(directions, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)
        
    def test_render_rays(self):
        """Test ray rendering."""
        batch_size = 2
        height = 32
        width = 32
        
        # Create dummy inputs
        ray_origins = torch.zeros(batch_size, height, width, 3)
        ray_directions = torch.zeros(batch_size, height, width, 3)
        ray_directions[..., 2] = -1  # Looking along negative z-axis
        
        latent_features = torch.randn(batch_size, 32, 8, 8)
        
        # Render rays
        rendered = self.model.render_rays(
            ray_origins, ray_directions, latent_features
        )
        
        # Check output shape
        assert rendered.shape == (batch_size, height, width, 3)
        
        # Check value range
        assert torch.all(rendered >= 0) and torch.all(rendered <= 1)
        
    def test_forward(self):
        """Test forward pass of full model."""
        # Create dummy input
        images = torch.randn(2, 3, 64, 64)
        
        # Run forward pass
        output = self.model(images)
        
        # Check output shape
        assert output.shape == (2, 64, 64, 3)
        
        # Check value range
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test that model can run on GPU."""
        # Move model to GPU
        model_gpu = self.model.to('cuda')
        
        # Create dummy input on GPU
        images = torch.randn(2, 3, 64, 64, device='cuda')
        
        # Run forward pass
        output = model_gpu(images)
        
        # Check output
        assert output.device.type == 'cuda'
        assert output.shape == (2, 64, 64, 3)
