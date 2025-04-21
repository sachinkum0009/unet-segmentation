import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_dataset = CarvanaDataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = CarvanaDataset(val_img_dir, val_mask_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # Add channel dimension
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (preds.sum() + y.sum() + 1e-8)
    acc = num_correct / num_pixels
    print(f"Got {num_correct} / {num_pixels} with accuracy {acc}")
    print(f"Dice Score: {dice_score / len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # torchvision.utils.save_image(x, f"{folder}/x_{idx}.png")
            torchvision.utils.save_image(preds, f"{folder}/preds_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/y_{idx}.png")

    model.train()
