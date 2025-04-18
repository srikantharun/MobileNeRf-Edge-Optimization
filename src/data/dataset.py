"""
Dataset management for MobileNeRF Edge.
"""
import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from .preprocessing import ImagePreprocessor


class SceneDataset(Dataset):
    """
    Dataset for training and evaluating scene reconstruction models.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 transform: Optional[Any] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 max_samples: Optional[int] = None):
        """
        Initialize the scene dataset.
        
        Args:
            data_dir: Directory containing scene images and metadata
            split: Dataset split ('train', 'val', or 'test')
            transform: Additional transforms to apply
            target_size: Target image size
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load scene metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._generate_metadata()
            
        # Get samples for the specified split
        split_file = self.data_dir / f'{split}_files.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            self.file_list = self._create_split()
            
        # Limit number of samples if specified
        if max_samples is not None and max_samples < len(self.file_list):
            self.file_list = self.file_list[:max_samples]
            
        # Initialize preprocessor
        camera_matrix = None
        dist_coeffs = None
        if 'camera_matrix' in self.metadata and 'dist_coeffs' in self.metadata:
            camera_matrix = np.array(self.metadata['camera_matrix'])
            dist_coeffs = np.array(self.metadata['dist_coeffs'])
            
        self.preprocessor = ImagePreprocessor(
            target_size=target_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
        )
    
    def _generate_metadata(self) -> Dict:
        """
        Generate metadata for the dataset if none exists.
        
        Returns:
            Dictionary of metadata
        """
        # Get all image files
        image_files = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))
        image_files = [str(f.relative_to(self.data_dir)) for f in image_files]
        
        metadata = {
            'num_images': len(image_files),
            'image_files': image_files,
            'camera_matrix': None,  # To be filled by calibration
            'dist_coeffs': None,    # To be filled by calibration
            'scene_scale': 1.0,
            'scene_center': [0, 0, 0]
        }
        
        # Save metadata
        with open(self.data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
    
    def _create_split(self) -> List[str]:
        """
        Create train/val/test splits if they don't exist.
        
        Returns:
            List of files for the current split
        """
        image_files = self.metadata['image_files']
        
        # Shuffle files with fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        # Create splits (80% train, 10% val, 10% test)
        n_total = len(image_files)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if self.split == 'train':
            files = image_files[:n_train]
        elif self.split == 'val':
            files = image_files[n_train:n_train+n_val]
        else:  # test
            files = image_files[n_train+n_val:]
            
        # Save the split
        with open(self.data_dir / f'{self.split}_files.txt', 'w') as f:
            f.write('\n'.join(files))
            
        return files
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        # Get image path
        img_path = str(self.data_dir / self.file_list[idx])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        img = self.preprocessor.process(img)
        
        # Apply any additional transforms
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            
        # Get the camera pose if available
        camera_idx = self._get_camera_idx(self.file_list[idx])
        
        pose = None
        if 'poses' in self.metadata and camera_idx is not None:
            pose_data = self.metadata['poses'][camera_idx]
            rotation = torch.tensor(pose_data['rotation'], dtype=torch.float32)
            translation = torch.tensor(pose_data['translation'], dtype=torch.float32)
            pose = {'rotation': rotation, 'translation': translation}
        
        # Prepare sample data
        sample = {
            'image': img,
            'file_path': img_path,
        }
        
        # Add pose information if available
        if pose is not None:
            sample['pose'] = pose
            
        return sample
    
    def _get_camera_idx(self, file_path: str) -> Optional[int]:
        """
        Get the camera index for a file path.
        
        Args:
            file_path: Image file path
            
        Returns:
            Camera index if available, None otherwise
        """
        if 'image_files' not in self.metadata:
            return None
            
        # Find the index of the file in the metadata
        try:
            return self.metadata['image_files'].index(file_path)
        except ValueError:
            return None


def create_dataloader(data_dir: Union[str, Path],
                     split: str = 'train',
                     batch_size: int = 8,
                     num_workers: int = 4,
                     target_size: Tuple[int, int] = (256, 256),
                     transform: Optional[Any] = None,
                     max_samples: Optional[int] = None) -> DataLoader:
    """
    Create a dataloader for the scene dataset.
    
    Args:
        data_dir: Directory containing scene images and metadata
        split: Dataset split ('train', 'val', or 'test')
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Target image size
        transform: Additional transforms to apply
        max_samples: Maximum number of samples to use
        
    Returns:
        DataLoader for the specified dataset
    """
    dataset = SceneDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        target_size=target_size,
        max_samples=max_samples
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
