import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from skimage import io
from src.data.synthetic_data_generation import generate_synthetic_dataset


class SyntheticChocolateDataset(Dataset):

    def __init__(
        self,
        background_dir: str,
        alpha_reference_dir: str,
        synth_dir: str,
        original_label_csv: str,
        train_idx, # tensor of indices
        per_background: int = 2,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        
        # saving the parameters
        self.alpha_reference_dir = Path(alpha_reference_dir)
        self.background_dir = Path(background_dir)
        self.synth_dir = Path(synth_dir)
        self.per_background = per_background
        self.transform = transform
        self.target_transform = target_transform
        # convert the tensor to a list
        self.train_idx = train_idx.indices
        
        background_df = pd.read_csv(original_label_csv)
        background_df = background_df.iloc[self.train_idx].reset_index(drop=True)
        
        print("Generating the synthetic images in the background directory...")
        print(f"Using {len(self.train_idx)} as background images")
        self.synth_csv = generate_synthetic_dataset(per_background, self.background_dir, alpha_reference_dir, self.synth_dir, background_df, noise=True)
        
        self.dataset_len = len(self.synth_csv)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        
        # In case the index is a tensor, we convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reconstruct the image path with the image ID (/!\ L prefix)
        img_path = Path(f"{self.synth_dir}/L{self.synth_csv.iloc[idx, 0]}.JPG")

        image = io.imread(img_path)
        label = self.synth_csv.iloc[idx, 1:] # Skip the image ID
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
    