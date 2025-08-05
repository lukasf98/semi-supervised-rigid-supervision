from torch.utils.data import Dataset
import os
import json
import itk
import torch
import numpy as np
from multiprocessing import Manager
from pathlib import Path


class NLSTDataset(Dataset):
    """
    NLST Dataset for medical image registration.
    Loads paired medical images with their corresponding labels and rigid transformation data.
    """
    
    def __init__(self, data_dir, rigid_data_dir, ts_data_dir, stage='train', 
                 use_cache=True, shared_cache=True):
        """
        Initialize NLST Dataset.
        
        Args:
            data_dir (str): Directory containing NLST_dataset.json and image data
            rigid_data_dir (str): Directory containing rigid transformation data
            ts_data_dir (str): Directory containing tissue segmentation data
            stage (str): 'train' or 'val' for dataset split
            use_cache (bool): Whether to cache loaded data in memory
            shared_cache (bool): Whether to share cache across DataLoader workers
        """
        self.data_dir = Path(data_dir)
        self.rigid_data_dir = Path(rigid_data_dir)
        self.ts_data_dir = Path(ts_data_dir)
        self.stage = stage
        
        # Load dataset configuration
        self._load_dataset_config()
        
        # Initialize caching
        self.use_cache = use_cache
        self._setup_cache(shared_cache)
    
    def _load_dataset_config(self):
        """Load and parse the dataset configuration JSON file."""
        data_json = self.data_dir / "NLST_dataset.json"
        
        if not data_json.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {data_json}")
        
        with open(data_json) as file:
            data = json.load(file)
            
            # Parse training pairs
            train_files = []
            for pair in data["training_paired_images"]:
                # Extract filename components
                nam_fixed = Path(pair["fixed"]).stem
                nam_moving = Path(pair["moving"]).stem
                nam_fixed2 = nam_fixed.split("_")[1]
                nam_moving2 = nam_moving.split("_")[1]
                
                # Build file paths
                file_dict = self._build_file_paths(
                    nam_fixed, nam_moving, nam_fixed2, nam_moving2
                )
                train_files.append(file_dict)
            
            # Split dataset
            self._split_dataset(train_files)
    
    def _build_file_paths(self, nam_fixed, nam_moving, nam_fixed2, nam_moving2):
        """Build dictionary of file paths for a single data pair."""
        return {
            # Original images
            "fixed_image": self.data_dir / "imagesTr" / f"{nam_fixed}_0000_0000.nii.gz",
            "moving_image": self.data_dir / "imagesTr" / f"{nam_moving}_0000_0000.nii.gz",
            
            # Labels/masks
            "fixed_label": self.data_dir / "masksTr" / f"{nam_fixed}.nii.gz",
            "moving_label": self.data_dir / "masksTr" / f"{nam_moving}.nii.gz",
            
            # Keypoints
            "fixed_keypoints": self.data_dir / "keypointsTr" / f"{nam_fixed}.csv",
            "moving_keypoints": self.data_dir / "keypointsTr" / f"{nam_moving}.csv",
            
            # Rigid transformation labels
            "fixed_rigid_label": self.rigid_data_dir / f"{nam_moving2}_0001_to_0000" / "0000_rigid_masks_combined.nii.gz",
            "moving_rigid_label": self.rigid_data_dir / f"{nam_moving2}_0001_to_0000" / "0001_rigid_masks_combined.nii.gz",
            
            # Tissue segmentation labels
            "fixed_lv_label": self.ts_data_dir / f"{nam_fixed}.nii",
            "moving_lv_label": self.ts_data_dir / f"{nam_moving}.nii",
            
            # Rigid transformation data (from zip files)
            "rigid_00001_to_0000_label": self.rigid_data_dir / f"{nam_moving2}_0001_to_0000" / "rigid_target_mask.npy",
            "rigid_00000_to_0001_label": self.rigid_data_dir / nam_moving2 / "rigid_target_mask.npy",
            "rigid_00001_to_0000_ddf": self.rigid_data_dir / f"{nam_moving2}_0001_to_0000" / "rigid_mask.npy",
            "rigid_00000_to_0001_ddf": self.rigid_data_dir / nam_moving2 / "rigid_mask.npy",
        }
    
    def _split_dataset(self, train_files):
        """Split dataset into train/validation/test sets."""
        total_files = len(train_files)
        split_idx1 = int(total_files * 0.8)
        split_idx2 = int(total_files * 0.9)
        
        train_split = train_files[:split_idx1]
        val_split = train_files[split_idx1:split_idx2]
        test_split = train_files[split_idx2:]
        
        if self.stage == 'train':
            self.data_dicts = train_split
        elif self.stage == 'val':
            self.data_dicts = val_split
        elif self.stage == 'test':
            self.data_dicts = test_split
        else:
            raise ValueError(f"Unknown stage: {self.stage}. Use 'train', 'val', or 'test'.")
        
        print(f"Loaded {len(self.data_dicts)} {self.stage} samples")
    
    def _setup_cache(self, shared_cache):
        """Initialize caching system."""
        if self.use_cache:
            if shared_cache:
                # Use Manager dict for sharing across DataLoader workers
                self._cache = Manager().dict()
            else:
                self._cache = {}
        else:
            self._cache = None
    
    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, idx):
        """Load and return a single data sample."""
        # Check cache first
        if self.use_cache and idx in self._cache:
            return self._cache[idx]
        
        data_item = self.data_dicts[idx]
        
        # Load medical images
        sample_data = self._load_sample_data(data_item)
        
        # Cache the result
        if self.use_cache:
            self._cache[idx] = sample_data
        
        return sample_data
    
    def _load_sample_data(self, data_item):
        """Load all data for a single sample."""
        # Load ITK images
        fixed_image_orig = self._load_itk_image(data_item['fixed_image'])
        moving_image_orig = self._load_itk_image(data_item['moving_image'])
        fixed_label = self._load_itk_image(data_item['fixed_label'])
        moving_label = self._load_itk_image(data_item['moving_label'])
        fixed_rigid_label = self._load_itk_image(data_item['fixed_rigid_label'])
        moving_rigid_label = self._load_itk_image(data_item['moving_rigid_label'])
        fixed_lv_label = self._load_itk_image(data_item['fixed_lv_label'])
        moving_lv_label = self._load_itk_image(data_item['moving_lv_label'])
        
        # Load numpy arrays (rigid transformation data)
        rigid_data = self._load_rigid_data(data_item)
        
        # Apply intensity scaling to images
        fixed_image = self._scale_intensity_range(
            fixed_image_orig, a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True
        )
        moving_image = self._scale_intensity_range(
            moving_image_orig, a_min=-1200, a_max=400, b_min=0.0, b_max=1.0, clip=True
        )
        
        # Add channel dimension
        images_and_labels = [
            fixed_image, moving_image, fixed_image_orig, moving_image_orig,
            fixed_label, moving_label, fixed_rigid_label, moving_rigid_label,
            fixed_lv_label, moving_lv_label
        ]
        
        for i, data in enumerate(images_and_labels):
            images_and_labels[i] = data[None, ...]  # Add channel dimension
        
        # Create sample dictionary
        sample = {
            'fixed_image': torch.tensor(images_and_labels[0], dtype=torch.float32),
            'moving_image': torch.tensor(images_and_labels[1], dtype=torch.float32),
            'fixed_image_orig': torch.tensor(images_and_labels[2], dtype=torch.float32),
            'moving_image_orig': torch.tensor(images_and_labels[3], dtype=torch.float32),
            'fixed_label': torch.tensor(images_and_labels[4], dtype=torch.float32),
            'moving_label': torch.tensor(images_and_labels[5], dtype=torch.float32),
            'fixed_rigid_label': torch.tensor(images_and_labels[6], dtype=torch.float32),
            'moving_rigid_label': torch.tensor(images_and_labels[7], dtype=torch.float32),
            'fixed_lv_label': torch.tensor(images_and_labels[8], dtype=torch.float32),
            'moving_lv_label': torch.tensor(images_and_labels[9], dtype=torch.float32),
            
            # Keypoint file paths (loaded as needed)
            'fixed_keypoints': str(data_item['fixed_keypoints']),
            'moving_keypoints': str(data_item['moving_keypoints']),
            
            # Rigid transformation data
            'rigid_00001_to_0000_ddf': torch.tensor(rigid_data['ddf_1_to_0'], dtype=torch.float32),
            'rigid_00000_to_0001_ddf': torch.tensor(rigid_data['ddf_0_to_1'], dtype=torch.float32),
            'rigid_00001_to_0000_label': torch.tensor(rigid_data['label_1_to_0'], dtype=torch.float32),
            'rigid_00000_to_0001_label': torch.tensor(rigid_data['label_0_to_1'], dtype=torch.float32),
            
            # Reference for metadata
            'reference': str(data_item['fixed_image']),
        }
        
        return sample
    
    def _load_rigid_data(self, data_item):
        """Load rigid transformation data from .npy files."""
        try:
            rigid_data = {
                'ddf_1_to_0': np.load(data_item["rigid_00001_to_0000_ddf"]),
                'ddf_0_to_1': np.load(data_item["rigid_00000_to_0001_ddf"]),
                'label_1_to_0': np.load(data_item["rigid_00001_to_0000_label"]),
                'label_0_to_1': np.load(data_item["rigid_00000_to_0001_label"]),
            }
            return rigid_data
        except FileNotFoundError as e:
            print(f"Warning: Could not load rigid data: {e}")
            # Return zero arrays as fallback
            dummy_shape = (64, 64, 64)  # Adjust based on your data
            return {
                'ddf_1_to_0': np.zeros(dummy_shape, dtype=np.float32),
                'ddf_0_to_1': np.zeros(dummy_shape, dtype=np.float32),
                'label_1_to_0': np.zeros(dummy_shape, dtype=np.float32),
                'label_0_to_1': np.zeros(dummy_shape, dtype=np.float32),
            }
    
    def _load_itk_image(self, file_path):
        """Load ITK image and convert to numpy array."""
        try:
            image = itk.imread(str(file_path))
            image_np = itk.array_view_from_image(image)
            return image_np
        except Exception as e:
            print(f"Warning: Could not load image {file_path}: {e}")
            # Return dummy data as fallback
            return np.zeros((64, 64, 64), dtype=np.float32)
    
    @staticmethod
    def _scale_intensity_range(img, a_min, a_max, b_min, b_max, clip):
        """Scale image intensity from [a_min, a_max] to [b_min, b_max]."""
        img = (img - a_min) / (a_max - a_min)
        img = img * (b_max - b_min) + b_min
        if clip:
            img = np.clip(img, b_min, b_max)
        return img
    
    def get_sample_info(self, idx):
        """Get information about a sample without loading the data."""
        if idx >= len(self.data_dicts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        data_item = self.data_dicts[idx]
        return {
            'index': idx,
            'fixed_image_path': str(data_item['fixed_image']),
            'moving_image_path': str(data_item['moving_image']),
            'stage': self.stage,
        }
    
    def verify_files(self, verbose=True):
        """Verify that all required files exist."""
        missing_files = []
        
        for idx, data_item in enumerate(self.data_dicts):
            for key, file_path in data_item.items():
                if isinstance(file_path, (str, Path)) and not Path(file_path).exists():
                    missing_files.append((idx, key, file_path))
        
        if verbose:
            if missing_files:
                print(f"Found {len(missing_files)} missing files:")
                for idx, key, path in missing_files[:10]:  # Show first 10
                    print(f"  Sample {idx}, {key}: {path}")
                if len(missing_files) > 10:
                    print(f"  ... and {len(missing_files) - 10} more")
            else:
                print("All files verified successfully!")
        
        return len(missing_files) == 0