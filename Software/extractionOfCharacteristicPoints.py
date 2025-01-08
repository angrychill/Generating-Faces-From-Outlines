import cv2
import dlib
import os
from pathlib import Path
import numpy as np

# Define paths for processed train and validation data
PROCESSED_TRAIN_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/PROCESSED_TRAIN_DATA"
PROCESSED_VALID_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/PROCESSED_VALID_DATA"

# Define paths for saving new data
NEW_TRAIN_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/NEW_TRAIN_FOLDER"
NEW_VALID_FOLDER = "C:/Users/Administrator/Downloads/DatasetBS/archive12/lfw-funneled/NEW_VALID_FOLDER"

# Ensure the output directories exist
Path(NEW_TRAIN_FOLDER).mkdir(parents=True, exist_ok=True)
Path(NEW_VALID_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model from Dlib's website

# Function to extract and save facial landmarks and create annotated images
def process_images_with_landmarks(input_folder, output_folder, target_size=(256, 256)):
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.jpeg')]

    print(f"Processing images from {input_folder}...")

    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        # Resize image to target size
        img_resized = cv2.resize(img, target_size)

        # Create a blank white image of the same size
        blank_image = np.ones_like(img_resized) * 255

        # Convert to grayscale for dlib
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Draw circles for eyes, nose, and mouth
            points = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))
                cv2.circle(blank_image, (x, y), 2, (0, 0, 0), -1)

            # Draw additional shapes for eyes, nose, and mouth
            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            mouth = [landmarks.part(i) for i in range(48, 68)]
            nose = [landmarks.part(i) for i in range(27, 36)]

            for eye in [left_eye, right_eye]:
                for point in eye:
                    cv2.circle(blank_image, (point.x, point.y), 2, (0, 0, 0), -1)

            for point in mouth:
                cv2.circle(blank_image, (point.x, point.y), 2, (0, 0, 0), -1)

            for point in nose:
                cv2.circle(blank_image, (point.x, point.y), 2, (0, 0, 0), -1)

            # Optionally save points to a file
            points_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_points.txt")
            with open(points_file, "w") as f:
                for point in points:
                    f.write(f"{point[0]}, {point[1]}\n")

        # Save the new annotated image
        output_img_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_img_path, blank_image)

    print(f"Processed images saved in {output_folder}")

# Process train and validation datasets
process_images_with_landmarks(PROCESSED_TRAIN_FOLDER, NEW_TRAIN_FOLDER)
process_images_with_landmarks(PROCESSED_VALID_FOLDER, NEW_VALID_FOLDER)
