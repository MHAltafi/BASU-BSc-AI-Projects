import os
import cv2
import bm3d
import lpips
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

import tkinter as tk
from tkinter import filedialog

import warnings
warnings.filterwarnings("ignore")

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

noisy_folder = filedialog.askdirectory(title="Select the folder containing noisy images")

original_folder = filedialog.askdirectory(title="Select the folder containing clean images")

dest_folder = filedialog.askdirectory(title="Select the folder to save denoised images")

classifier_path = filedialog.askopenfilename(title="Select the Phase 2 ViT classifier", filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")])

periodic_model_path = filedialog.askopenfilename(title="Select the periodic autoencoder checkpoint", filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")])

csv_path = filedialog.asksaveasfilename(title="Save results CSV", initialfile="DenoisingResults.csv", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

root.destroy()

if not all([noisy_folder, original_folder, dest_folder, classifier_path, periodic_model_path, csv_path]):
    raise ValueError("Selection cancelled. Run this cell again and select all paths.")

os.makedirs(dest_folder, exist_ok=True)

# print("Noisy folder:", noisy_folder)
# print("Clean folder:", original_folder)
# print("Output folder:", dest_folder)
# print("Classifier:", classifier_path)
# print("Periodic model:", periodic_model_path)
# print("Results CSV:", csv_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier_model = torch.load(classifier_path, map_location=device, weights_only=False)

classifier_model = classifier_model.to(device)
classifier_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.55, 0.55, 0.55],
        std=[0.22, 0.22, 0.22]
    )
])


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PeriodicAutoencoder(nn.Module):

    def __init__(self):
        super(PeriodicAutoencoder, self).__init__()

        # Encoder
        self.encoder1 = ConvBlock(1, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder4 = ConvBlock(1024 + 512, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder3 = ConvBlock(512 + 256, 256)

        self.up2 = nn.Upsample( scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder2 = ConvBlock(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder1 = ConvBlock(128 + 64, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        output = torch.sigmoid(self.output(d1))

        return output


periodic_model = PeriodicAutoencoder().to(device)

periodic_model.load_state_dict(torch.load(periodic_model_path, map_location=device, weights_only=True))

periodic_model.eval()

print("Models loaded successfully on:", device)


def gaussian_denoiser(image, c=30):
    image_float = image.astype(np.float32) / 255.0

    denoised = bm3d.bm3d(image_float, sigma_psd=c / 255.0)

    return np.uint8(np.clip(denoised * 255.0, 0, 255))


def salt_and_pepper_denoiser(image):
    first = cv2.medianBlur(image, 3)
    second = cv2.medianBlur(first, 3)
    return second


def periodic_denoiser(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    resized_image = cv2.resize(gray_image, (256, 256), interpolation=cv2.INTER_AREA)

    input_tensor = torch.from_numpy(resized_image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = periodic_model(input_tensor)

    denoised_image = output.squeeze().cpu().numpy()
    denoised_image = np.uint8(np.clip(denoised_image * 255.0, 0, 255))

    denoised_image = cv2.resize(denoised_image, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return denoised_image


def classifier(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier_model(input_tensor)

    predicted_idx = torch.argmax(output, dim=1).item()

    return predicted_idx


def denoiser(image):
    predicted_class = classifier(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if predicted_class == 0:
        return gaussian_denoiser(gray_image), 'Gaussian'

    elif predicted_class == 1:
        return salt_and_pepper_denoiser(gray_image), 'Salt'

    elif predicted_class == 2:
        return periodic_denoiser(gray_image), 'Periodic'

    else:
        raise ValueError(f"Unexpected predicted class: {predicted_class}")


def psnr_calculator(original, image):
    return cv2.PSNR(original.astype(np.float64), image.astype(np.float64))


def ssim_calculator(original_img, denoised_img):
    return ssim(original_img, denoised_img, data_range=255)


alexnet_lpips = lpips.LPIPS(net='alex').to(device)
alexnet_lpips.eval()


def lpips_calculator(original_img, denoised_img):
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    denoised_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB)

    original_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1

    denoised_tensor = torch.from_numpy(denoised_rgb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1

    with torch.no_grad():
        score = alexnet_lpips(original_tensor.to(device), denoised_tensor.to(device))

    return score.item()


def load_images_from_folder_and_denoise(noisy, original, dest):
    results = []

    os.makedirs(dest, exist_ok=True)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    filenames = sorted([filename for filename in os.listdir(noisy) if filename.lower().endswith(extensions)])

    for filename in filenames:
        noisy_path = os.path.join(noisy, filename)
        original_path = os.path.join(original, filename)

        img = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
        img_orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Could not read noisy image:", noisy_path)
            continue

        if img_orig is None:
            print("Matching clean image not found:", original_path)
            continue

        denoised, noise_type = denoiser(img)

        height, width = img_orig.shape

        if denoised.shape != img_orig.shape:
            denoised = cv2.resize(denoised, (width, height), interpolation=cv2.INTER_LINEAR)

        psnr_value = psnr_calculator(img_orig, denoised)
        ssim_value = ssim_calculator(img_orig, denoised)
        lpips_value = lpips_calculator(img_orig, denoised)

        cv2.imwrite(os.path.join(dest, filename), denoised)

        results.append({
            'Image Name': filename,
            'Noise Type': noise_type,
            'PSNR': psnr_value,
            'SSIM': ssim_value,
            'LPIPS': lpips_value
        })

    return pd.DataFrame(results)


results = load_images_from_folder_and_denoise(noisy_folder, original_folder, dest_folder)

print(results.to_string(index=False))


results.to_csv(csv_path, index=False)

print("Images processed:", len(results))

if not results.empty:
    print("Mean PSNR:", results['PSNR'].mean())
    print("Mean SSIM:", results['SSIM'].mean())
    print("Mean LPIPS:", results['LPIPS'].mean())

    print(results.groupby('Noise Type')[['PSNR', 'SSIM', 'LPIPS']].mean())
else:
    print("No matching image pairs were processed.")