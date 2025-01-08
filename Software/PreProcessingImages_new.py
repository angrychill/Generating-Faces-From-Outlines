import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

TRAIN_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/TRAIN_DATA"
VALID_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/VALID_DATA"
PROCESSED_TRAIN_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/PROCESSED_TRAIN_DATA"
PROCESSED_VALID_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/PROCESSED_VALID_DATA"

BATCH_SIZE = 128  # Broj slika koje procesiramo odjednom
TARGET_SIZE = (256, 256)  # Ciljna veličina slika (256x256)

# Kreiraj foldere za obrađene slike
os.makedirs(PROCESSED_TRAIN_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VALID_FOLDER, exist_ok=True)

def adjust_contrast(image, alpha=1.2, beta=30):
    """Povećaj kontrast slike."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def rotate_image(image, angle):
    """Rotiraj sliku za zadani kut."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def flip_image(image):
    """Okreni sliku horizontalno."""
    return cv2.flip(image, 1)

def apply_transformations(image):
    """Primijeni blagu transformaciju na sliku (rotacija, flip, kontrast)."""
    # Promijeni kontrast
    image = adjust_contrast(image)
    
    # Rotacija za slučajni kut između -15 i 15 stupnjeva
    angle = np.random.uniform(-15, 15)
    image = rotate_image(image, angle)

    # Slučajni flip
    if np.random.rand() > 0.5:
        image = flip_image(image)

    return image

def resize_image(image, target_size):
    """Promijeni veličinu slike na ciljanu veličinu."""
    return cv2.resize(image, target_size)

def preprocess_image(image_path, output_folder):
    """Predprocesiraj jednu sliku."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    # Uklanjanje šuma koristeći Gaussov filter
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Poboljšanje oštrine slike koristeći Unsharp Masking
    image_blurred = cv2.GaussianBlur(image, (21, 21), 10)
    image = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    # Primijeni dodatne transformacije
    image = apply_transformations(image)

    # Promijeni veličinu slike
    image = resize_image(image, TARGET_SIZE)

    # Spremi obrađenu sliku
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)
    print(f"Slika spremljena: {output_path}")

def preprocess_images_in_batches(input_folder, output_folder, batch_size):
    """Funkcija za obrađivanje slika u batchovima."""
    all_image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                all_image_paths.append(image_path)

    # Podijeli slike na batchove
    batches = [all_image_paths[i:i + batch_size] for i in range(0, len(all_image_paths), batch_size)]

    # Paralelno obradi batchove koristeći višestruko procesiranje
    with ProcessPoolExecutor() as executor:
        for batch in batches:
            executor.map(preprocess_image, batch, [output_folder] * len(batch))

if __name__ == '__main__':
    # Predprocesiraj slike u treniranim i validacijskim folderima
    preprocess_images_in_batches(TRAIN_FOLDER, PROCESSED_TRAIN_FOLDER, BATCH_SIZE)
    preprocess_images_in_batches(VALID_FOLDER, PROCESSED_VALID_FOLDER, BATCH_SIZE)

    print(f"Predprocesirane slike spremljene u: {PROCESSED_TRAIN_FOLDER} i {PROCESSED_VALID_FOLDER}")
