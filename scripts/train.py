#!/usr/bin/env python
"""
Training script for MobileNeRF Edge.
"""
import argparse
import yaml
import os
import torch
import torch.nn as nn
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from src.data.dataset import create_dataloader
from src.models.mobilenerf import MobileNeRF, load_pretrained_mobilenerf, fine_tune_mobilenerf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MobileNeRF Edge model')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--pretrained-model', type=str, help='Path to pretrained model')
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device to use (cuda, cpu)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.pretrained_model:
        config['model']['pretrained_path'] = args.pretrained_model
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['training']['device'] = args.device
    
    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir=config['data']['data_dir'],
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['data']['target_size']),
        max_samples=config['training']['max_samples'] if args.debug else None
    )
    
    val_loader = create_dataloader(
        data_dir=config['data']['data_dir'],
        split='val',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        target_size=tuple(config['data']['target_size']),
        max_samples=config['training']['max_samples'] if args.debug else None
    )
    
    # Load or create model
    if config['model']['pretrained_path'] and os.path.exists(config['model']['pretrained_path']):
        print(f"Loading pretrained model from {config['model']['pretrained_path']}")
        model = load_pretrained_mobilenerf(config['model']['pretrained_path'], device)
    else:
        print("Creating new model")
        model = MobileNeRF(
            latent_dim=config['model']['latent_dim'],
            base_channels=config['model']['base_channels'],
            num_samples=config['model']['num_samples'],
            pos_encoding_freqs=config['model']['pos_encoding_freqs'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers']
        )
    
    # Move model to device
    model = model.to(device)
    
    # Fine-tune model
    print("Starting fine-tuning...")
    model = fine_tune_mobilenerf(
        model=model,
        train_loader=train_loader,
        num_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        device=device
    )
    
    # Evaluate model
    print("Evaluating model...")
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, images.permute(0, 2, 3, 1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.6f}")
    
    # Save model
    model_path = output_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Log metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"Validation Loss: {avg_val_loss:.6f}\n")
    
    print("Training complete!")


if __name__ == '__main__':
    main()
