import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

class SyntheticChocolateDataset(Dataset):
    """
    Much faster PyTorch Dataset for generating synthetic choc images:
      - preloads/resizes all reference PNGs once
      - caches background paths & labels in plain NumPy/torch
      - minimal pandas use (only in __init__)
    """
    def __init__(
        self,
        background_dir: str,
        alpha_reference_dir: str,
        original_label_csv: str,
        train_idx: list,
        per_background: int = 2,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        bg_dir = Path(background_dir)
        self.per_background = per_background
        self.transform = transform
        self.target_transform = target_transform

        # 1) Load label DataFrame ONCE
        df = pd.read_csv(original_label_csv)
        # extract only the rows we’ll use, and the numeric labels (drop ID col)
        df_train = df.iloc[train_idx].reset_index(drop=True)
        self.labels = df_train.iloc[:, 1:].to_numpy(dtype=np.int32)

        # 2) Cache list of background image paths (matching L<ID>.JPG)
        self.bg_paths = [
            bg_dir / f"L{int(r[0])}.JPG"
            for r in df_train.itertuples(index=False, name=None)
        ]

        # 3) Preload + resize all reference chocolate PNGs
        ref_dir = Path(alpha_reference_dir)
        self.choco_keys = [
            "Jelly White", "Jelly Milk", "Jelly Black", "Amandina",
            "Crème brulée", "Triangolo", "Tentation noir", "Comtesse",
            "Noblesse", "Noir authentique", "Passion au lait",
            "Arabia", "Stracciatella"
        ]

        self.base_chocolates = []
        for name in self.choco_keys:
            fn = (
                name.lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("é", "e")
                    .replace("è", "e")
                    .replace("'", "_")
                + ".png"
            )
            img = Image.open(ref_dir / fn).convert("RGBA")
            # resize *once* to the target dataset image size:
            # assume all backgrounds share same size → inspect first
            target_size = Image.open(self.bg_paths[0]).size
            img = img.resize(target_size, Image.LANCZOS)
            self.base_chocolates.append(img)

        # Precompute Dataset length
        self.dataset_len = len(self.bg_paths) * self.per_background

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # map linear idx → which background, which repeat
        bg_idx = (idx // self.per_background)
        label = self.labels[bg_idx].copy()  # shape = (n_choc_types,)

        # 1) load bg once per sample
        bg = Image.open(self.bg_paths[bg_idx]).convert("RGBA")
        bg_w, bg_h = bg.size

        # 2) prepare a mask for collision‐checking
        mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

        # 3) paste a few chocolates
        counts = np.zeros(len(self.base_chocolates), dtype=np.int32)
        n_to_paste = np.random.randint(1, 4)
        for _ in range(n_to_paste):
            i = np.random.randint(len(self.base_chocolates))
            base = self.base_chocolates[i]

            # apply random flip/rotate *in-memory only*
            ch = base
            if torch.rand(1).item() < 0.5:
                ch = ImageOps.mirror(ch)
            if torch.rand(1).item() < 0.5:
                ch = ImageOps.flip(ch)
            angle = int(torch.randint(0, 360, (1,)).item())
            ch = ch.rotate(angle, expand=False)

            # brute‐force random spot until it fits
            ch_w, ch_h = ch.size
            alpha = np.array(ch.split()[-1])
            placed = False
            for _ in range(100):  # cap attempts
                x = np.random.randint(0, bg_w - ch_w + 1)
                y = np.random.randint(0, bg_h - ch_h + 1)
                region = mask[y: y + ch_h, x: x + ch_w]
                if not np.any((alpha > 0) & (region > 0)):
                    # safe to paste
                    mask[y: y + ch_h, x: x + ch_w] = np.maximum(region, alpha)
                    bg.paste(ch, (x, y), ch)
                    counts[i] += 1
                    placed = True
                    break
            # if no spot found in 100 tries, skip

        # 4) convert bg → RGB, combine original & new counts
        synth = bg.convert("RGB")
        label += counts

        # 5) transforms
        if self.transform:
            synth = self.transform(synth)
        if self.target_transform:
            label = self.target_transform(label)

        return synth, torch.from_numpy(label).int()


class LabelToTensor:
    """
    Convert a label (pandas Series) to a tensor
    """
    def __call__(self, label):
        return torch.tensor(label.to_numpy())
    