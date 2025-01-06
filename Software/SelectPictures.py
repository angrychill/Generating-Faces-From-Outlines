import os
import shutil
import random

MAIN_FOLDER = "PATH"
TRAIN_FOLDER = "PATH"
VALID_FOLDER = "PATH"

TRAIN_SPLIT = 0.7

os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VALID_FOLDER, exist_ok=True)

def get_all_images(main_folder):
    """Funkcija za dobivanje svih slika iz svih podfoldera."""
    image_paths = []
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

def split_images(image_paths, train_split):
    """Funkcija za podjelu slika na TRAIN i VALID setove."""
    random.shuffle(image_paths)
    split_index = int(len(image_paths) * train_split)
    return image_paths[:split_index], image_paths[split_index:]

def save_images(image_paths, destination_folder):
    """Funkcija za spremanje slika u zadani folder."""
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy(image_path, destination_path)

all_images = get_all_images(MAIN_FOLDER)

train_images, valid_images = split_images(all_images, TRAIN_SPLIT)

save_images(train_images, TRAIN_FOLDER)
save_images(valid_images, VALID_FOLDER)

print(f"Ukupan broj slika: {len(all_images)}")
print(f"Broj slika u TRAIN_DATA: {len(train_images)}")
print(f"Broj slika u VALID_DATA: {len(valid_images)}")