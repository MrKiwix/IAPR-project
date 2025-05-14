# Data loading for the test dataset

from torch.utils.data import Dataset
from skimage import io
from pathlib import Path
import torch
import os

class ChocolateTestDataset(Dataset):
    """Dataset for test images where we don't have labels -> we cannot use the same as for training"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing all the test images
            transform (torchvision.transforms, optional):  Transformation or sequence of transformation to apply to the input images. Defaults to None.
        """        
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        # Greedily fetch all .JPG files in the testing directory
        self.image_files = [f for f in os.listdir(data_dir)]
        
        # we sort the files to ensure a consistent order (in training we don't really care)
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        # In case the index is a tensor, we convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.image_files[idx]
        img_path = Path(f"{self.data_dir}/{img_name}")
        
        # We want to follow the same norm as training, so we remove the 'L' prefix and put only the ID
        image_id = img_name.split('.')[0]
        image_id = image_id[1:]  # Remove L
            
        image = io.imread(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and its ID in the csv
        return image, image_id