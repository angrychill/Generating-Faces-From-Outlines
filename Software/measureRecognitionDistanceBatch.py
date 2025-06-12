import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from deepface import DeepFace
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
from pathlib import Path
import csv

from PIL import Image
import numpy as np


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Subset

from pix2pix.models import *
from pix2pix.datasets import *

from sklearn.metrics.pairwise import cosine_similarity

BASE_FOLDER = "C:/Users/Iris/Documents/TrainingData New/"
MAIN_FOLDER = BASE_FOLDER + "celeb" # koji dataset koristiti

GENERATOR_MODEL_FILE = "C:/Users/Iris/Documents/Generating-Faces-From-Outlines/saved_models/outlines/generator_40.pth"
DISCRIMINATOR_MODEL_FILE = "C:/Users/Iris/Documents/Generating-Faces-From-Outlines/saved_models/outlines/discriminator_40.pth"

SET_OUTPUT = "C:/Users/Iris/Documents/RI_Training_Data/FinalValidationSet"
SET_INPUT = "C:/Users/Iris/Documents/RI_Training_Data/ValidationSet"

VALIDATION_OUTPUT_DIR = "C:/Users/Iris/Documents/RI_Training_Data/ComparisonOutput"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

facenet_model = DeepFace.build_model("Facenet")


# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()
#print("checkpoint 5")
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

#load model!
generator.load_state_dict(torch.load(GENERATOR_MODEL_FILE))
discriminator.load_state_dict(torch.load(DISCRIMINATOR_MODEL_FILE))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

dataloader = DataLoader(
    ImageDataset(SET_INPUT, SET_OUTPUT, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)


full_dataset = ImageDataset(SET_INPUT, SET_OUTPUT, transforms_=transforms_, mode="val")
subset_dataset = Subset(full_dataset, list(range(10)))

val_dataloader = DataLoader(
    subset_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

def tensor_to_pil(img_tensor):
    """
    Convert a torch tensor image to PIL Image.
    img_tensor shape: (C, H, W), normalized [-1, 1]
    """
    img_np = img_tensor.cpu().detach().numpy()
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # scale [-1,1] to [0,255]
    img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W to H,W,C
    return Image.fromarray(img_np)

def precompute_real_embeddings(val_dataloader, model):
    real_embeddings = []
    print("Precomputing real embeddings...")
    for i, imgs in enumerate(val_dataloader):
        real_img_tensor = imgs["A"][0]  # Ground truth face image tensor (batch_size=1)
        pil_img = tensor_to_pil(real_img_tensor)
        img_np = np.array(pil_img)  # <-- Convert PIL to numpy here!
        emb = DeepFace.represent(img_np, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        real_embeddings.append(emb)
    print("Done precomputing real embeddings.")
    return real_embeddings

def batch_compute_fake_embeddings(fake_imgs_tensor_list, model):
    pil_imgs = [tensor_to_pil(img_tensor[0]) for img_tensor in fake_imgs_tensor_list]  # list of PIL images
    embeddings = []
    for pil_img in pil_imgs:
        img_np = np.array(pil_img)  # Convert PIL to numpy array
        emb = DeepFace.represent(img_np, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        embeddings.append(emb)
    return embeddings




def compute_similarity(img1_path, img2_path, model):
    # Get embeddings
    emb1 = DeepFace.represent(img_path=img1_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    emb2 = DeepFace.represent(img_path=img2_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]

    # Cosine similarity
    return cosine_similarity([emb1], [emb2])[0][0]


def get_face_comp_result(val_dataloader, generator, output_dir, real_embeddings, output_file="similarities.csv"):
    os.makedirs(output_dir, exist_ok=True)
    similarities = []
    fake_images = []

    generator.eval()

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Similarity"])

        # Step 1: Generate all fake images first
        for i, imgs in enumerate(val_dataloader):
            real_A = imgs["B"].type(Tensor)  # Outline image
            real_B = imgs["A"].type(Tensor)  # Ground truth face

            with torch.no_grad():
                fake_B = generator(real_A)

            fake_images.append(fake_B.cpu())

            # Save combined image for visual comparison (optional)
            combined = make_grid(
                [real_A.data[0].cpu(), fake_B.data[0].cpu(), real_B.data[0].cpu()],
                nrow=3,
                normalize=True,
                padding=10
            )
            combined_path = os.path.join(output_dir, f"comparison_{i}.png")
            save_image(combined, combined_path)

        # Step 2: Batch compute fake embeddings
        fake_embeddings = batch_compute_fake_embeddings(fake_images, facenet_model)

        # Step 3: Compute cosine similarity for each pair
        for i, fake_emb in enumerate(fake_embeddings):
            try:
                real_emb = real_embeddings[i]
                similarity = cosine_similarity([fake_emb], [real_emb])[0][0]
            except Exception as e:
                print(f"Error computing similarity for index {i}: {e}")
                similarity = 0.0

            similarities.append(similarity)
            writer.writerow([i, similarity])

    return similarities


## THE HIGHER THE NUMBER, THE MORE SIMILAR THE FACE
## -1 : TOTALLY DISSIMILAR, 1 : TOTALLY SIMILAR

if __name__ == "__main__":
    # Precompute embeddings for all real images in val set
    real_embeddings = precompute_real_embeddings(val_dataloader, facenet_model)

    # Path to directory with all 40 generator models
    generator_models_dir = "C:/Users/Iris/Documents/Generating-Faces-From-Outlines/saved_models/outlines"
    
    # Output CSV file to store average similarities per epoch
    summary_csv = os.path.join(generator_models_dir, "epoch_similarities.csv")
    
    # Prepare summary output CSV
    with open(summary_csv, mode="w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["Epoch", "Average Similarity"])
        
        for epoch in range(1, 41):  # Assuming generator_1.pth to generator_40.pth
            gen_model_path = os.path.join(generator_models_dir, f"generator_{epoch}.pth")
            
            if not os.path.exists(gen_model_path):
                print(f"Model not found for epoch {epoch}: {gen_model_path}")
                continue
            
            print(f"Evaluating epoch {epoch}...")
            
            # Load the generator weights for this epoch
            generator.load_state_dict(torch.load(gen_model_path))
            generator.eval()
            
            # Output directory for images per epoch
            epoch_output_dir = os.path.join(VALIDATION_OUTPUT_DIR, f"epoch_{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            
            # File to save similarities for each image pair
            sim_file = os.path.join(epoch_output_dir, f"similarities_epoch_{epoch}.csv")
            
            # Evaluate similarities using the new function
            similarities = get_face_comp_result(val_dataloader, generator, epoch_output_dir, real_embeddings, sim_file)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # Log result
            writer.writerow([epoch, avg_similarity])
            print(f"Epoch {epoch} - Average Similarity: {avg_similarity:.4f}")
