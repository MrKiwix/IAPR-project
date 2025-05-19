# Dataset class for the Chocolate validation dataset

from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
from src.data.synthetic_data_generation import generate_synthetic_image
from PIL import Image

class SyntheticChocolateDataset(Dataset):
    """
    Pytorch Dataset for the Chocolate validation dataset
    """    
    
    def __init__(self, background_dir, alpha_reference_dir, original_label_csv, per_background = 3, transform=None, target_transform=None):
        """
        Args:
            background_dir (str): Path to the directory containing all the background images
            alpha_reference_dir (str): Path to the directory containing all the alpha reference images
            original_label_csv (str): Path to the CSV file containing the origin labels
            per_background (int): Number of images to generate per background. Defaults to 3.
            transform (callable, optional):  Transformation or sequence of transformation to apply to the input images. Defaults to None.
            target_transform (callable, optional):  Transformation or sequence of transformation to apply to the target labels. Defaults to None.
        """        
        
        super().__init__()
        self.background_dir = background_dir
        self.label_df = pd.read_csv(original_label_csv)
        self.reference_dir = alpha_reference_dir
        self.transform = transform
        self.target_transform = target_transform
        self.per_background = per_background
        self.df_size = len(self.label_df)

    def __len__(self):
        return self.per_background * self.df_size

    def __getitem__(self, idx):
        
        # In case the index is a tensor, we convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reconstruct the image path with the image ID (/!\ L prefix)
        background_path = Path(f"{self.background_dir}/L{self.label_df.iloc[idx % self.df_size, 0]}.JPG") # we loop over the dataset

        # Open the image using PIL for RGBA utilities
        bg = Image.open(background_path).convert("RGBA")
        original_label = self.label_df.iloc[idx % self.df_size, 1:].astype(int) # Skip the image ID

        # Now, we want to generate the synthetic image
        synth, new_label = generate_synthetic_image(bg, self.reference_dir, bg.size, False)
        
        # We then need to merge the two labels
        new_label = new_label.values() # it's a dict so we get the values
        new_label = pd.Series(new_label, index=self.label_df.columns[1:])
        new_label = new_label.astype(int)
        
        # Merge the two labels
        new_label = original_label + new_label
        
        # Apply the transformations if needed
        if self.transform:
            synth = self.transform(synth)
        if self.target_transform:
            new_label = self.target_transform(new_label)

        return synth, new_label

class LabelToTensor:
    """
    Convert a label (pandas Series) to a tensor
    """
    def __call__(self, label):
        return torch.tensor(label.to_numpy())
    