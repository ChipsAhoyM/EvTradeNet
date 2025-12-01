import os
import zipfile
from huggingface_hub import hf_hub_download


repo_id = "ChipsAhoyMG/ERD"
files = ["blur.zip", "event.zip", "sharp.zip"]

local_zips = {}

for fname in files:
    print(f"Downloading {fname} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        repo_type="dataset",
        local_dir="."
    )
    local_zips[fname] = path


target_dir = "EvRGB_Deblur"
os.makedirs(target_dir, exist_ok=True)


def unzip_to(path, out_dir):
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(out_dir)

print("\nUnzipping files...")
for fname, zip_path in local_zips.items():
    unzip_to(zip_path, target_dir)
    print(f"Unzipped {fname} into {target_dir}/")
    # Remove the zip file after successful extraction to save space
    try:
        os.remove(zip_path)
        print(f"Deleted {fname} ({zip_path})")
    except OSError as e:
        print(f"Warning: could not delete {fname} ({zip_path}): {e}")

print("\nDone! Your dataset structure is now like:")
print("""
EvRGB_Deblur/
  blur/*.png
  event/*.txt
  sharp/*.png
""")