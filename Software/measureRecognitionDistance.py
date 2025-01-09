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

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from pix2pix.models import *
from pix2pix.datasets import *

BASE_FOLDER = "C:/Users/Iris/Documents/TrainingData New/"
MAIN_FOLDER = BASE_FOLDER + "celeb" # koji dataset koristiti

GENERATOR_MODEL_FILE = "C:/Program Files/GitHub projects/Facial-Deidentification-Using-GAN/Software/pix2pix/saved_models/faces_deid_celeb/generator_8.pth"
DISCRIMINATOR_MODEL_FILE = "C:/Program Files/GitHub projects/Facial-Deidentification-Using-GAN/Software/pix2pix/saved_models/faces_deid_celeb/discriminator_8.pth"

SET_OUTPUT = "C:/Users/Iris/Documents/TrainingData New/celebExtracted_Validation_Data"
SET_INPUT = "C:/Users/Iris/Documents/TrainingData New/celebCombined_Validation_Set"

VALIDATION_OUTPUT_DIR = MAIN_FOLDER + "validation_output"

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

val_dataloader = DataLoader(
    ImageDataset(SET_INPUT, SET_OUTPUT, transforms_=transforms_, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

def get_face_comp_result(val_dataloader, generator, output_dir, output_file="similarities_epoch_8.csv"):
    os.makedirs(output_dir, exist_ok=True)
    similarities = []
   
    generator.eval()
    
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Similarity"])
        
        for i, imgs in enumerate(val_dataloader):
            # input A, target B
            real_A = Variable(imgs["B"].type(Tensor))  # Extracted facial characteristics
            real_B = Variable(imgs["A"].type(Tensor))
        
        # generation of reconstructed face
            fake_B = generator(real_A)
            fake_path = os.path.join(output_dir, f"fake_{i}.png")
            real_path = os.path.join(output_dir, f"real_{i}.png")
        
            save_image(fake_B.data, fake_path, normalize=True)
            save_image(real_B.data, real_path, normalize=True)
        # distance_metric (string): Metric for measuring similarity. Options: 'cosine' 'euclidean', 'euclidean_l2' 
            result_cos = DeepFace.verify(fake_path, real_path, model_name="Facenet", distance_metric="cosine")
            similarity = 1 - result_cos["distance"]
            
            
            similarities.append(similarity)
            
            writer.writerow([i, similarity])
    
    return similarities

if __name__ == "__main__":
    output_dir = VALIDATION_OUTPUT_DIR
    similarities = get_face_comp_result(val_dataloader, generator, output_dir, output_dir + "similarities.csv")

    avg_similarity = sum(similarities) / len(similarities)
    print(f"Average Similarity: {avg_similarity:.4f}")