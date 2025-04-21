import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = 'data/train'
masks_dir = 'data/train_masks'

# Output folders
train_images_out = 'dataset/train_imgs'
train_masks_out = 'dataset/train_masks_split'
val_images_out = 'dataset/val_imgs'
val_masks_out = 'dataset/val_masks'

# Create output directories
os.makedirs(train_images_out, exist_ok=True)
os.makedirs(train_masks_out, exist_ok=True)
os.makedirs(val_images_out, exist_ok=True)
os.makedirs(val_masks_out, exist_ok=True)

# Get all image filenames
all_images = os.listdir(images_dir)

# Split
train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Move files
def move_files(file_list, src_dir, dst_dir):
    for file_name in file_list:
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        shutil.copy(src, dst)

def move_mask_files(file_list, src_dir, dst_dir):
    for file_name in file_list:
        file_name = file_name.replace('.jpg', '_mask.gif')  # Ensure the file extension is correct
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)
        shutil.copy(src, dst)

# Move training images and masks
move_files(train_files, images_dir, train_images_out)
move_mask_files(train_files, masks_dir, train_masks_out)

# Move validation images and masks
move_files(val_files, images_dir, val_images_out)
move_mask_files(val_files, masks_dir, val_masks_out)