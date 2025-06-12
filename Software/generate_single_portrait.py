import torch
from torchvision import transforms
from PIL import Image
import os
from pix2pix import GeneratorUNet  # Make sure this import works in your environment
from torchvision.utils import save_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, img_height=256, img_width=256):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)

def generate(input_path, output_path, model_path, img_height=256, img_width=256):
    # Initialize generator and load weights
    generator = GeneratorUNet()
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()

    # Load input image
    input_img = load_image(input_path, img_height, img_width)

    # Generate output
    with torch.no_grad():
        fake_img = generator(input_img)
    
    # Denormalize and save
    fake_img = 0.5 * (fake_img + 1.0)  # from [-1,1] to [0,1]
    save_image(fake_img, output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("output_image", type=str, help="Path to save generated image")
    parser.add_argument("model_path", type=str, help="Path to trained generator model .pth file")
    parser.add_argument("--img_height", type=int, default=256, help="Image height")
    parser.add_argument("--img_width", type=int, default=256, help="Image width")
    args = parser.parse_args()

    generate(args.input_image, args.output_image, args.model_path, args.img_height, args.img_width)
