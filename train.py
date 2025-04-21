import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim

from model import UNet
from dataset import CarvanaDataset

from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy, save_predictions_as_imgs

# Hyperparameters

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 240 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train_imgs"
TRAIN_MASK_DIR = "dataset/train_masks_split"
TEST_IMG_DIR = "dataset/test_images"
TEST_MASK_DIR = "dataset/test_masks"
VAL_IMG_DIR = "dataset/val_imgs"
VAL_MASK_DIR = "dataset/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.unsqueeze(1).to(DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update tqdm
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # cross entroy loss for multi-class
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        val_img_dir=VAL_IMG_DIR,
        val_mask_dir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder=f"saved_images/{epoch}", device=DEVICE
        )
        

if __name__ == "__main__":
    main()