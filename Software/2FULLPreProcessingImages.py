import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE_FOLDER = "C:/Users/Iris/Documents/RI_Training_Data/"

TRAIN_FOLDER = BASE_FOLDER + "TrainingSet"
VALID_FOLDER = BASE_FOLDER + "ValidationSet"


FINAL_TRAIN_FOLDER = BASE_FOLDER + "FinalTrainingSet"
FINAL_VALID_FOLDER = BASE_FOLDER + "FinalValidationSet"

BATCH_SIZE = 128  # Broj slika koje procesiramo odjednom
TARGET_SIZE = (256, 256)  # Ciljna veličina slika (256x256)

# Kreiraj foldere za obrađene slike
os.makedirs(FINAL_TRAIN_FOLDER, exist_ok=True)
os.makedirs(FINAL_VALID_FOLDER, exist_ok=True)

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

def resize_image(image, target_size):
    """Promijeni veličinu slike na ciljanu veličinu."""
    return cv2.resize(image, target_size)

def preprocess_image_step1(image_path, output_folder):
    """Prvi korak predprocesiranja: promijeni kontrast, izoštri i promijeni veličinu."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    # Uklanjanje šuma koristeći Gaussov filter
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # Poboljšanje oštrine slike koristeći Unsharp Masking
    # image_blurred = cv2.GaussianBlur(image, (21, 21), 10)
    # image = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    # Promijeni kontrast
    # image = adjust_contrast(image)

    # Promijeni veličinu slike
    # image = resize_image(image, TARGET_SIZE)

    # Spremi obrađenu sliku
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    # output_path_combined = os.path.join(combined_folder, "step_2_", filename)
    cv2.imwrite(output_path, image)
    # cv2.imwrite(output_path_combined, image)
    print(f"Slika spremljena u STEP1: {output_path}")

def preprocess_image_step2(image_path, output_folder):
    """Drugi korak predprocesiranja: rotacija i flip."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    # Rotacija za slučajni kut između -15 i 15 stupnjeva
    angle = np.random.uniform(-15, 15)
    image = rotate_image(image, angle)

    # Slučajni flip
    if np.random.rand() > 0.5:
        image = flip_image(image)

    # Spremi obrađenu sliku
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    # output_path_combined = os.path.join(combined_folder, "step_2_", filename)
    cv2.imwrite(output_path, image)
    # cv2.imwrite(output_path_combined, image)
    # print(f"Slika spremljena u STEP2: {output_path} i {output_path_combined}")

def preprocess_image_step3(image_path, output_folder):
    """Kreiranje outlines"""
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    # Pretvori sliku u sivu skalu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Primijeni GaussianBlur za smanjenje šuma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Primijeni Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Invertiraj boje kako bi rubovi bili crni na bijeloj pozadini
    outline = cv2.bitwise_not(edges)

    # Spremi konturiranu sliku
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, outline)

def sobel_edge_detection(
    image_path,
    output_folder,
    ksize=3,
    ddepth=cv2.CV_64F,
    threshold_value=30,
    invert=True,
    scale=1,
    delta=-0.25
    ):
    """Performs Sobel edge detection with fine-tuning options."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Sobel gradients
    sobelx = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta)
    sobely = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta)

    # Magnitude of gradient
    sobel = np.hypot(sobelx, sobely)

    # Normalize to 0-255 and convert to uint8
    sobel = np.uint8(np.clip((sobel / np.max(sobel)) * 255, 0, 255))

    # Thresholding (binarization)
    _, edge = cv2.threshold(sobel, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert if desired
    if invert:
        edge = cv2.bitwise_not(edge)

    # Save result
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, edge)
    # print(f"Sobel konture spremljene: {output_path}")



def preprocess_images_in_batches(input_folder, output_folder, batch_size, step_function):
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
            executor.map(step_function, batch, [output_folder] * len(batch))

if __name__ == '__main__':
    # Prvi korak predprocesiranja: promijeni kontrast, izoštri i promijeni veličinu
    # preprocess_images_in_batches(TRAIN_FOLDER, PROCESSED_TRAIN_FOLDER_STEP1, BATCH_SIZE, preprocess_image_step1)
    # preprocess_images_in_batches(VALID_FOLDER, PROCESSED_VALID_FOLDER_STEP1, BATCH_SIZE, preprocess_image_step1)

    # # Drugi korak predprocesiranja: rotacija i flip
    # preprocess_images_in_batches(PROCESSED_TRAIN_FOLDER_STEP1, PROCESSED_TRAIN_FOLDER_STEP2, BATCH_SIZE, preprocess_image_step2)
    # preprocess_images_in_batches(PROCESSED_VALID_FOLDER_STEP1, PROCESSED_VALID_FOLDER_STEP2, BATCH_SIZE, preprocess_image_step2)

    preprocess_images_in_batches(TRAIN_FOLDER, FINAL_TRAIN_FOLDER, BATCH_SIZE, sobel_edge_detection)
    preprocess_images_in_batches(VALID_FOLDER, FINAL_VALID_FOLDER, BATCH_SIZE, sobel_edge_detection)
    print(f"Predprocesirane slike spremljene u foldere.")
