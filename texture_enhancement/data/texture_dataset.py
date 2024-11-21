import torch
from basicsr.data.base_dataset import BaseDataset
from basicsr.utils.registry import DATASET_REGISTRY
from pathlib import Path
import cv2
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

@DATASET_REGISTRY.register()
class TextureDataset(BaseDataset):
    """Texture dataset for neural texture enhancement.
    
    Read RGB image, depth map, normal map, UV map and ground truth.
    The data structure should be or left to be further changed:
        root
        ├── sample1_rgb.png
        ├── sample1_depth.png
        ├── sample1_normal.png
        ├── sample1_uv.png
        ├── sample1_gt.png
        ├── sample2_rgb.png
        └── ...
    """
    
    def __init__(self, opt):
        """Initialize the texture dataset.
        
        Args:
            opt (dict): Config for train datasets. It contains the following keys:
                dataroot (str): Data root path
                gt_size (int): Ground truth image size
                use_hflip (bool): Whether to horizontally flip
                use_rot (bool): Whether to rotate
                phase (str): 'train' or 'val'
        """
        super(TextureDataset, self).__init__(opt)
        self.data_root = Path(opt['dataroot'])
        # Find all RGB images and assume other modalities follow the same naming pattern
        self.sample_list = list(self.data_root.glob('*_rgb.png'))
        
        # Image size settings
        self.gt_size = opt.get('gt_size', 256)
        self.input_size = self.gt_size // opt.get('scale', 4)
        
        # Data augmentation settings
        self.use_flip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        
        # Mean and std for normalization
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.sample_list)

    def __getitem__(self, index):
        """Get training data.
        
        Args:
            index (int): Index

        Returns:
            dict: Includes:
                lq: Low quality input (RGB + depth + normal + UV)
                gt: Ground truth
                lq_path: Path to input image
        """
        # Get image paths
        rgb_path = self.sample_list[index]
        sample_name = rgb_path.stem.replace('_rgb', '')
        
        # Read all modalities
        # RGB input - BGR to RGB conversion and normalization
        rgb = cv2.imread(str(rgb_path))[...,::-1] / 255.0
        
        # Depth map - single channel
        depth_path = rgb_path.parent / f'{sample_name}_depth.png'
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE) / 255.0
        
        # Normal map - BGR to RGB conversion and normalization
        normal_path = rgb_path.parent / f'{sample_name}_normal.png'
        normal = cv2.imread(str(normal_path))[...,::-1] / 255.0
        
        # UV map - only use U and V channels
        uv_path = rgb_path.parent / f'{sample_name}_uv.png'
        uv = cv2.imread(str(uv_path))[...,::-1][:, :, :2] / 255.0
        
        # Ground truth
        gt_path = rgb_path.parent / f'{sample_name}_gt.png'
        gt = cv2.imread(str(gt_path))[...,::-1] / 255.0

        # Random crop
        if self.opt['phase'] == 'train':
            # Random crop
            h, w = rgb.shape[:2]
            x = np.random.randint(0, max(0, w - self.gt_size))
            y = np.random.randint(0, max(0, h - self.gt_size))
            
            # Crop all modalities
            rgb = rgb[y:y+self.gt_size, x:x+self.gt_size]
            depth = depth[y:y+self.gt_size, x:x+self.gt_size]
            normal = normal[y:y+self.gt_size, x:x+self.gt_size]
            uv = uv[y:y+self.gt_size, x:x+self.gt_size]
            gt = gt[y:y+self.gt_size, x:x+self.gt_size]

            # Random flip and rotation
            if self.use_flip and np.random.rand() < 0.5:
                rgb = rgb[:, ::-1, :]
                depth = depth[:, ::-1]
                normal = normal[:, ::-1, :]
                uv = uv[:, ::-1, :]
                gt = gt[:, ::-1, :]
                
            if self.use_rot and np.random.rand() < 0.5:
                # 90-degree rotation
                rgb = np.rot90(rgb, k=1)
                depth = np.rot90(depth, k=1)
                normal = np.rot90(normal, k=1)
                uv = np.rot90(uv, k=1)
                gt = np.rot90(gt, k=1)

        # Combine input modalities
        input_tensor = np.concatenate([
            rgb,                    # RGB channels
            depth[..., None],       # Depth channel
            normal,                 # Normal channels
            uv                      # UV channels
        ], axis=2)
        
        # To tensor and normalize
        input_tensor = torch.from_numpy(input_tensor.transpose(2, 0, 1)).float()
        gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        
        # Normalize RGB channels of input and GT
        input_tensor[0:3] = normalize(input_tensor[0:3], self.mean, self.std)
        gt_tensor = normalize(gt_tensor, self.mean, self.std)

        return {
            'lq': input_tensor,  # Low quality input
            'gt': gt_tensor,     # Ground truth
            'lq_path': str(rgb_path),  # Path to input image
            'gt_path': str(gt_path)    # Path to ground truth image
        }

    def __repr__(self):
        """Print dataset information."""
        return (f'{self.__class__.__name__} Dataset with {self.__len__()} samples - '
                f'Data root: {self.data_root}')