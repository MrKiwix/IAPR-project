# Dataset class for the Chocolate training dataset

from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from pathlib import Path
import torch

class ChocolateDataset(Dataset):
    """
    Pytorch Dataset for the Chocolate training dataset
    """    
    
    def __init__(self, data_dir, label_csv, transform=None, target_transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the validation images
            label_csv (str): Path to the csv file containing the labels and image IDs
            transform (torchvision.transforms, optional): Transformation or sequence of transformation to apply to the input images. Defaults to None.
            target_transform (torchvision.transforms, optional): Transformation or sequence of transformation to apply to the labels. Defaults to None.
        """        
        
        super().__init__()
        self.data_dir = data_dir
        self.label_df = pd.read_csv(label_csv)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        
        # In case the index is a tensor, we convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reconstruct the image path with the image ID (/!\ L prefix)
        img_path = Path(f"{self.data_dir}/L{self.label_df.iloc[idx, 0]}.JPG")

        # Read the picture from disk ONLY in the __getitem__ method for performance reasons (not in __init__)
        image = io.imread(img_path)
        label = self.label_df.iloc[idx, 1:] # Skip the image ID
        label = label.astype(int)

        # Apply the transformations if needed
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class LabelToTensor:
    """
    Convert a label (pandas Series) to a tensor
    """
    def __call__(self, label):
        return torch.tensor(label.to_numpy())
    