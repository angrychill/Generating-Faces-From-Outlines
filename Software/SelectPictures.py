import os
import shutil
import random
from deepface import DeepFace

BASE_FOLDER = "C:/Users/Iris/Documents/TrainingData/"
MAIN_FOLDER = BASE_FOLDER + "lfw_funneled"
TRAIN_FOLDER = BASE_FOLDER + "TrainingSet"
VALID_FOLDER = BASE_FOLDER + "ValidationSet"
MULTIPLE_FACES_FOLDER = BASE_FOLDER + "MultipleFaces"
NO_FACE_FOLDER = BASE_FOLDER + "NoFaces"

TRAIN_SPLIT = 0.8  # Omjer podjele slika na trening i validaciju
BATCH_SIZE = 128  # Broj slika koje procesiramo odjednom

# Kreiraj sve potrebne foldere
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(MULTIPLE_FACES_FOLDER, exist_ok=True)
os.makedirs(NO_FACE_FOLDER, exist_ok=True)

# Funkcija za dobivanje svih slika iz svih podfoldera
def get_all_images(main_folder):
    """Funkcija za dobivanje svih slika iz svih podfoldera."""
    image_paths = []
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Funkcija za podjelu slika na TRAIN i VALID setove
def split_images(image_paths, train_split):
    """Funkcija za podjelu slika na TRAIN i VALID setove."""
    random.shuffle(image_paths)
    split_index = int(len(image_paths) * train_split)
    return image_paths[:split_index], image_paths[split_index:]

# Funkcija za spremanje slika u zadani folder
def save_images(image_paths, destination_folder):
    """Funkcija za spremanje slika u zadani folder."""
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        destination_path = os.path.join(destination_folder, filename)
        
        # Provjera postoji li datoteka prije kopiranja
        if os.path.exists(image_path):
            shutil.copy(image_path, destination_path)
        else:
            print(f"Datoteka {image_path} nije pronađena.")

# Funkcija za klasifikaciju slika prema broju prepoznatih lica
def classify_and_move_images(image_paths):
    """Klasificira slike prema broju lica na njima i premješta ih u odgovarajuće foldere."""
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i:i + BATCH_SIZE]

        # Analiza slika za detekciju lica
        results = []
        for image_path in batch:
            try:
                # Detekcija lica pomoću extract_faces
                faces = DeepFace.extract_faces(image_path, enforce_detection=False)
                num_faces = len(faces)  # Broj prepoznatih lica
            except Exception as e:
                num_faces = 0

            results.append((image_path, num_faces))

        # Premještanje slika u odgovarajuće foldere
        for image_path, num_faces in results:
            filename = os.path.basename(image_path)
            if num_faces == 0:
                # Slika bez lica
                shutil.move(image_path, os.path.join(NO_FACE_FOLDER, filename))
            elif num_faces > 1:
                # Slika sa više od jednog lica
                shutil.move(image_path, os.path.join(MULTIPLE_FACES_FOLDER, filename))
            else:
                # Slika s jednim licem, premjestiti na odgovarajući folder
                destination_folder = TRAIN_FOLDER if random.random() < TRAIN_SPLIT else VALID_FOLDER
                shutil.move(image_path, os.path.join(destination_folder, filename))

# Glavni dio koda
all_images = get_all_images(MAIN_FOLDER)

train_images, valid_images = split_images(all_images, TRAIN_SPLIT)

# Klasifikacija slika prije premještanja u trening/validacijske foldere
classify_and_move_images(all_images)

# Klasifikacija za treniranje i validaciju
save_images(train_images, TRAIN_FOLDER)
save_images(valid_images, VALID_FOLDER)

print(f"Ukupan broj slika: {len(all_images)}")
print(f"Broj slika u TRAIN_DATA: {len(train_images)}")
print(f"Broj slika u VALID_DATA: {len(valid_images)}")
