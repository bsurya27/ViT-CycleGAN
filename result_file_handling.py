import os
import shutil

def move_real_fake_images(src_folder):
    real_folder = os.path.join(src_folder, "real_B")
    fake_folder = os.path.join(src_folder, "fake_B")
    os.makedirs(real_folder, exist_ok=True)
    os.makedirs(fake_folder, exist_ok=True)

    # Move files after checking source directory
    for fname in os.listdir(src_folder):
        fpath = os.path.join(src_folder, fname)
        if not os.path.isfile(fpath):
            continue  # skip folders

        if "real_B" in fname and not fname.startswith("real_B/"):
            shutil.copy(fpath, os.path.join(real_folder, fname))
        elif "fake_B" in fname and not fname.startswith("fake_B/"):
            shutil.copy(fpath, os.path.join(fake_folder, fname))

    print(f"✅ Images moved to: {real_folder} and {fake_folder}")
    return real_folder, fake_folder

# Paths
path_vit = 'results/final_results/water_cycleGAN/test_150/images'
path_base = 'results/final_results/vit_cycleGAN/test_150/images'

move_real_fake_images(path_vit)
move_real_fake_images(path_base)
