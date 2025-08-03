from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
from pytorch_fid import fid_score

def compute_psnr_ssim(real_folder, fake_folder):
    real_files = sorted([f for f in os.listdir(real_folder) if f.endswith('.png')])
    fake_files = sorted([f for f in os.listdir(fake_folder) if f.endswith('.png')])
    
    psnr_scores, ssim_scores = [], []

    for r, f in zip(real_files, fake_files):
        real = np.array(Image.open(os.path.join(real_folder, r)).convert("RGB"))
        fake = np.array(Image.open(os.path.join(fake_folder, f)).convert("RGB"))

        psnr_scores.append(psnr(real, fake, data_range=255))
        ssim_scores.append(ssim(real, fake, channel_axis=2, data_range=255))

    print(f"Average PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")



def compute_fid(real_folder, fake_folder, device='cuda'):
    fid = fid_score.calculate_fid_given_paths([real_folder, fake_folder], batch_size=32, device=device, dims=2048)
    print(f"FID: {fid:.2f}")

path_vit_fake = 'results/final_results/water_cycleGAN/test_150/images/fake_B'
path_vit_real = 'results/final_results/water_cycleGAN/test_150/images/real_B'


path_base_fake = 'results/final_results/vit_cycleGAN/test_150/images/fake_B'
path_base_real = 'results/final_results/vit_cycleGAN/test_150/images/real_B'

print("ViT-CycleGAN")
compute_psnr_ssim(path_vit_real, path_vit_fake)
compute_fid(path_vit_real, path_vit_fake)
print("CycleGAN")
compute_psnr_ssim(path_base_real, path_base_fake)
compute_fid(path_base_real, path_base_fake)
