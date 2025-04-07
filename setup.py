import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests


# ===== File Download Utilities =====
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded = 0

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                percent = int(100 * downloaded / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\r[DOWNLOAD] {percent}% complete")
                sys.stdout.flush()

        sys.stdout.write("\n")
        return True
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


# ===== Main Setup Function =====
def main():
    print("\n" + "=" * 60)
    print("YOLO-TinyPerson Environment Setup")
    print("=" * 60 + "\n")

    # ----- Install Required Packages -----
    try:
        import ultralytics

        print(f"[✓] Ultralytics version {ultralytics.__version__} is already installed")
    except ImportError:
        print("[*] Installing ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("[✓] Ultralytics installed successfully")

    from ultralytics import settings

    # ----- Configure Dataset Directory -----
    if os.name == "nt":
        dataset_dir = Path("E:/GitHub/YOLO-TinyPerson/dataset")
        font_dir = Path(os.path.expanduser("~")) / ".config" / "Ultralytics"
    else:
        dataset_dir = Path("/mnt/workspace/dataset")
        font_dir = Path("/root/.config/Ultralytics")
    settings.update({"datasets_dir": str(dataset_dir)})
    print(f"[i] Dataset directory set to {dataset_dir}")

    dataset_path = Path(dataset_dir)
    if dataset_path.exists():
        print(f"[✓] Dataset directory already exists at {dataset_dir}")
    else:
        print(f"[!] Dataset directory not found at {dataset_dir}")

        # ----- Download Dataset if Needed -----
        dataset_archive = Path("dataset.7z")
        if not dataset_archive.exists():
            print("[*] Downloading dataset.7z...")
            dataset_url = "https://github.com/xixu-me/YOLO-TinyPerson/releases/download/dataset/dataset.7z"
            if download_file(dataset_url, dataset_archive):
                print("[✓] Dataset download completed")
            else:
                print("[!] Dataset download failed. Cannot continue setup.")
                return
        else:
            print(f"[i] Found existing dataset archive at {dataset_archive}")

        # ----- Extract Dataset Archive -----
        print("[*] Extracting dataset.7z...")
        try:
            try:
                import py7zr

                print("[✓] py7zr is already installed")
            except ImportError:
                print("[*] Installing py7zr package...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
                print("[✓] py7zr installed successfully")
                import py7zr

            with py7zr.SevenZipFile(dataset_archive, mode="r") as z:
                z.extractall()
            print("[✓] Dataset extracted successfully")

            # ----- Move Dataset to Correct Location -----
            extracted_dataset = Path("dataset")
            if extracted_dataset.exists() and str(extracted_dataset) != str(
                dataset_path
            ):
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extracted_dataset), str(dataset_path))
                print(f"[✓] Dataset moved to {dataset_path}")

        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return

    # ----- Download Required Font -----
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "Arial.ttf"

    if not font_path.exists():
        print(f"[*] Downloading Arial.ttf to {font_path}...")
        url = "https://ultralytics.com/assets/Arial.ttf"
        download_file(url, font_path)
        print(f"[✓] Arial.ttf downloaded successfully to {font_path}")
    else:
        print(f"[✓] Arial.ttf already exists at {font_path}")

    # ----- Set Up Weights Directory -----
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Weights directory set to {weights_dir}")

    # ----- Download Pre-trained Models -----
    yolo_models = ["yolo11n", "yolov8x", "yolov9e", "yolov10x", "yolo11x", "yolo12x"]
    base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

    for model in yolo_models:
        weight_file = f"{model}.pt"
        weight_path = weights_dir / weight_file

        if weight_path.exists():
            print(f"[✓] {weight_file} already exists at {weight_path}")
        else:
            print(f"[*] Downloading {weight_file}...")
            weight_url = f"{base_url}{weight_file}"
            if download_file(weight_url, weight_path):
                print(f"[✓] {weight_file} downloaded successfully to {weight_path}")
            else:
                print(f"[ERROR] Failed to download {weight_file}")

    # ----- Completion Message -----
    print("\n" + "=" * 60)
    print("[✓] Environment setup complete!")
    print("=" * 60)


# ===== Script Entry Point =====
if __name__ == "__main__":
    main()
