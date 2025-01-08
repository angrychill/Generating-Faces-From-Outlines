import os
from PIL import Image

BASE_FOLDER = "C:/Users/Iris/Documents/TrainingData/"
TRAIN_FOLDER = BASE_FOLDER + "TrainingSet"
VALID_FOLDER = BASE_FOLDER + "ValidationSet"
PROCESSED_TRAIN_FOLDER_STEP1 = BASE_FOLDER + "Training_Step_1"
PROCESSED_VALID_FOLDER_STEP1 = BASE_FOLDER + "Valid_Step_1"
PROCESSED_TRAIN_FOLDER_STEP2 = BASE_FOLDER + "Training_Step_2"
PROCESSED_VALID_FOLDER_STEP2 = BASE_FOLDER + "Valid_Step_2"
COMBINED_TRAIN_FOLDER = BASE_FOLDER + "Combined_Training_Set"
COMBINED_VALID_FOLDER = BASE_FOLDER + "Combined_Validation_Set"
EXTRACTED_TRAINING_FOLDER = BASE_FOLDER + "Extracted_Training_Data"
EXTRACTED_VALIDATION_FOLDER = BASE_FOLDER + "Extracted_Validation_Data"
TRAIN_FOLDER_WITHOUT_TXT = BASE_FOLDER + "Combined_Extract_Training_ImgOnly"
VALID_FOLDER_WITHOUT_TXT = BASE_FOLDER + "Combined_Extract_Validation_ImgOnly"
TRAIN_FOLDER_WITH_TXT = BASE_FOLDER + "Combined_Extract_Training_TxtOnly"
VALID_FOLDER_WITH_TXT = BASE_FOLDER + "Combined__Extract_Validation_TxtOnly"

os.makedirs(TRAIN_FOLDER_WITHOUT_TXT, exist_ok=True)
os.makedirs(VALID_FOLDER_WITHOUT_TXT, exist_ok=True)
os.makedirs(TRAIN_FOLDER_WITH_TXT, exist_ok=True)
os.makedirs(VALID_FOLDER_WITH_TXT, exist_ok=True)

def extract_only_image_files(input_folder, output_folder):
    all_image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                all_image_paths.append(image_path)
    
    for path in all_image_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(path) as img:
                # Save the image to the output folder
                img.save(output_path)
            print(f"Saved {filename} to {output_path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")


def extract_only_text_files(input_folder, output_folder):
    all_text_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                image_path = os.path.join(root, file)
                all_text_paths.append(image_path)
    
    for path in all_text_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with open(path, 'r', encoding='utf-8') as infile:
                content = infile.read()
            
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(content)
            
            print(f"Saved {filename} to {output_path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")
    

extract_only_image_files(EXTRACTED_TRAINING_FOLDER, TRAIN_FOLDER_WITHOUT_TXT)
extract_only_image_files(EXTRACTED_VALIDATION_FOLDER, VALID_FOLDER_WITHOUT_TXT)
extract_only_text_files(EXTRACTED_TRAINING_FOLDER, TRAIN_FOLDER_WITH_TXT)
extract_only_text_files(EXTRACTED_VALIDATION_FOLDER, VALID_FOLDER_WITH_TXT)