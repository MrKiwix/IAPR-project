# This file will return a submission.csv file in the project's root directory containing the predicted labels for the test set.
# WARNING: It is assumed that the model weights are stored in the 'model' directory (best_choco_model.pt) and that the test set is stored in the 'data/test' directory.

# External Libraries
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import pandas as pd
from pathlib import Path
from tqdm import tqdm # for progress bar
import time

# Internal modules
from src.helper import *
from src.data.TestChocolateDataset import ChocolateTestDataset
from src.model.ChocoNetwork import ChocoNetwork

# Constants
MODEL_PATH = Path("model/best_choco_model.pt")
TESTING_DATA_PATH = Path("data/test")
OUTPUT_CSV_PATH = Path("submission.csv")
CLASS_NAMES = ["Jelly White","Jelly Milk","Jelly Black","Amandina","Crème brulée","Triangolo","Tentation noir","Comtesse","Noblesse","Noir authentique","Passion au lait","Arabia","Stracciatella"]
IMG_SIZE = (200, 300) # (height, width)
BATCH_SIZE = 32

# TRANSFORM DEFINITION
test_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(IMG_SIZE, antialias=True),
    v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

if __name__ == "__main__":
    
    # Check one what device we'll run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # define the dataset and create the dataloader
    test_dataset = ChocolateTestDataset(
        data_dir=TESTING_DATA_PATH,
        transform=test_tf
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    
    # Instantiate the model
    model = ChocoNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set the model to evaluation mode
    
    # Create the submission dataframe
    submission_df = pd.DataFrame(columns=["id"] + CLASS_NAMES)
    
    # Start the inference
    start_time = time.time()
    
    predictions = []
    ids = []
    with torch.no_grad(): # Disable gradient calculation -> speed up the process
        for images, image_ids in tqdm(test_loader, desc="Processing test images", unit="batch"):
            images = images.to(device, non_blocking=True)
            
            # Add the predictions to the list, cpu() is needed to get it back to the host memory
            predictions.extend(torch.round(model(images)).int().cpu().numpy())
            ids.extend(image_ids)
    
    # Insert in the dataframe       
    submission_df["ID"] = ids
    submission_df[CLASS_NAMES] = predictions
    
    # export as csv
    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
