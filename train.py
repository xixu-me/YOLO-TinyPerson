import time

import torch
from ultralytics import YOLO

# ===== Model Configuration =====
MODELS = [
    {"name": "YOLOv8x", "model_path": "weights/yolov8x.pt", "epochs": 2},
    {"name": "YOLOv9e", "model_path": "weights/yolov9e.pt", "epochs": 2},
    {"name": "YOLOv10x", "model_path": "weights/yolov10x.pt", "epochs": 2},
    {"name": "YOLO11x", "model_path": "weights/yolo11x.pt", "epochs": 2},
    {"name": "YOLO12x", "model_path": "weights/yolo12x.pt", "epochs": 2},
]


# ===== Model Training Function =====
def train_model(model_config, dataset_yaml="dataset/tinyperson.yaml"):
    print(f"\n{"="*60}")
    print(f"Training {model_config['name']} on TinyPerson dataset")
    print(f"{"="*60}\n")

    # ----- Load Model -----
    model = YOLO(model_config["model_path"])

    # ----- Configure Training Parameters -----
    hyperparams = {
        "data": dataset_yaml,
        "epochs": model_config["epochs"],
        "imgsz": 640,
        "batch": 0.90,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "lr0": 0.01,
        "lrf": 0.001,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "project": "results",
        "name": model_config["name"],
        "exist_ok": True,
        "patience": 50,
        "optimizer": "SGD",
        "cos_lr": True,
        "box": 7.5,
        "cls": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "scale": 0.3,
        "rect": False,
        "save": True,
        "save_period": 10,
    }

    # ----- Execute Training -----
    start_time = time.time()
    results = model.train(**hyperparams)

    # ----- Report Training Results -----
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n[✓] Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    output_path = f"results/{model_config['name']}/weights/best.pt"
    print(f"[i] Model saved to {output_path}")
    return str(output_path)


# ===== Main Training Execution =====
def main():
    trained_models = {}
    start_time = time.time()

    print("\n" + "=" * 60)
    print("YOLO-TinyPerson Training")
    print("=" * 60 + "\n")

    # ----- Check GPU Availability -----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[i] Training on GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        print(
            "[!] No GPU available. Training on CPU (not recommended for YOLO training)"
        )

    # ----- Train Each Model -----
    for model_config in MODELS:
        model_path = train_model(model_config)
        trained_models[model_config["name"]] = model_path

    # ----- Training Summary -----
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{"="*60}")
    print(f"[✓] All models trained in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{"="*60}")

    print("\n[i] Trained Models Summary:")
    print("-" * 60)
    for name, path in trained_models.items():
        print(f"[✓] {name}: {path}")
    print("-" * 60)


# ===== Script Entry Point =====
if __name__ == "__main__":
    main()
