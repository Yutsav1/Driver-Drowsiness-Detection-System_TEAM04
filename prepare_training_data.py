import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
datapath = "my_training_data"     # Folder where images/ and labels/ exist
outputpath = "data"               # Output folder for split data
train_ratio = 0.8                 # 80% training, 20% validation

# Create output folders
train_images = os.path.join(outputpath, "train/images")
train_labels = os.path.join(outputpath, "train/labels")
val_images = os.path.join(outputpath, "validation/images")
val_labels = os.path.join(outputpath, "validation/labels")

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Load dataset files
image_files = sorted(os.listdir(os.path.join(datapath, "images")))
label_files = sorted(os.listdir(os.path.join(datapath, "labels")))

# Split data
train_imgs, val_imgs = train_test_split(image_files, train_size=train_ratio, random_state=42)

# Move training files
for img in train_imgs:
    lbl = img.replace(".jpg", ".txt")

    shutil.copy(os.path.join(datapath, "images", img), train_images)
    shutil.copy(os.path.join(datapath, "labels", lbl), train_labels)

# Move validation files
for img in val_imgs:
    lbl = img.replace(".jpg", ".txt")

    shutil.copy(os.path.join(datapath, "images", img), val_images)
    shutil.copy(os.path.join(datapath, "labels", lbl), val_labels)

print("Dataset split complete! Training and validation folders created.")
