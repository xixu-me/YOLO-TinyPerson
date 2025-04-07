import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# ===== Model Configuration =====
MODELS = [
    {"name": "YOLOv8x", "model_path": "results/YOLOv8x/weights/best.pt"},
    {"name": "YOLOv9e", "model_path": "results/YOLOv9e/weights/best.pt"},
    {"name": "YOLOv10x", "model_path": "results/YOLOv10x/weights/best.pt"},
    {"name": "YOLO11x", "model_path": "results/YOLO11x/weights/best.pt"},
    {"name": "YOLO12x", "model_path": "results/YOLO12x/weights/best.pt"},
]

# ----- Color Scheme for Plots -----
COLORS = {
    "YOLOv8x": "#1f77b4",
    "YOLOv9e": "#ff7f0e",
    "YOLOv10x": "#2ca02c",
    "YOLO11x": "#d62728",
    "YOLO12x": "#9467bd",
}

# ----- Output Directories -----
EVAL_DIR = "evaluation"
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ===== Model Evaluation Function =====
def evaluate_model(model_config, val_data="dataset/images/val"):
    print(f"\n[*] Evaluating {model_config['name']} on TinyPerson test set...")

    try:
        # ----- Load Model -----
        model = YOLO(model_config["model_path"])

        # ----- Configure Evaluation Parameters -----
        eval_params = {
            "batch": 4,
            "conf": 0.25,
            "iou": 0.5,
            "max_det": 300,
            "device": 0 if torch.cuda.is_available() else "cpu",
        }

        # ----- Run Evaluation -----
        start_time = time.time()
        results = model.val(**eval_params)
        end_time = time.time()

        # ----- Calculate Metrics -----
        metrics = {
            "model": model_config["name"],
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
            "f1_score": 2
            * (
                results.results_dict["metrics/precision(B)"]
                * results.results_dict["metrics/recall(B)"]
            )
            / (
                results.results_dict["metrics/precision(B)"]
                + results.results_dict["metrics/recall(B)"]
                + 1e-10
            ),
            "mean_IoU": results.results_dict.get("metrics/iou(B)", 0),
            "inference_time": (end_time - start_time) / len(os.listdir(val_data)),
            "inference_fps": len(os.listdir(val_data)) / (end_time - start_time),
        }

        # ----- Get IoU if Available -----
        if hasattr(results, "box") and hasattr(results.box, "iou"):
            metrics["mean_IoU"] = results.box.iou.mean().item()

        print(
            f"[✓] Evaluation complete: Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}, "
            f"Inference Speed={metrics['inference_fps']:.2f} FPS"
        )

        return metrics

    except Exception as e:
        print(f"[ERROR] Evaluating {model_config['name']}: {str(e)}")
        return {
            "model": model_config["name"],
            "precision": 0,
            "recall": 0,
            "mAP50": 0,
            "mAP50-95": 0,
            "f1_score": 0,
            "mean_IoU": 0,
            "inference_time": 0,
            "inference_fps": 0,
            "error": str(e),
        }


# ===== Model Size Calculation Function =====
def measure_model_size(model_path):
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except:
        return 0


# ===== Plot Generation Function =====
def generate_plots(results_df):
    print("\n[*] Generating performance comparison plots...")

    # ----- Configure Plot Style -----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})

    # ----- Precision/Recall/F1 Bar Chart -----
    plt.figure(figsize=(12, 8))
    models = results_df["model"]
    x = np.arange(len(models))
    width = 0.25

    plt.bar(
        x - width, results_df["precision"], width, label="Precision", color="#3498db"
    )
    plt.bar(x, results_df["recall"], width, label="Recall", color="#2ecc71")
    plt.bar(x + width, results_df["f1_score"], width, label="F1-score", color="#e74c3c")

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-score Comparison")
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_f1.png"), dpi=300)
    print("[*] Created precision/recall/F1 comparison plot")

    # ----- mAP Bar Chart -----
    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, results_df["mAP50"], width, label="mAP@0.5", color="#9b59b6")
    plt.bar(
        x + width / 2,
        results_df["mAP50-95"],
        width,
        label="mAP@0.5:0.95",
        color="#f39c12",
    )

    plt.xlabel("Model")
    plt.ylabel("mAP")
    plt.title("mAP Comparison")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "map_comparison.png"), dpi=300)
    print("[*] Created mAP comparison plot")

    # ----- Inference Speed Bar Chart -----
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, results_df["inference_fps"], color="#1abc9c")

    plt.xlabel("Model")
    plt.ylabel("Frames Per Second (FPS)")
    plt.title("Inference Speed Comparison")
    plt.grid(axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "inference_speed.png"), dpi=300)
    print("[*] Created inference speed comparison plot")

    # ----- Precision vs Recall Scatter Plot -----
    plt.figure(figsize=(10, 8))

    for i, row in results_df.iterrows():
        model_name = row["model"]
        plt.scatter(
            row["recall"],
            row["precision"],
            s=row["mAP50"] * 500,
            color=COLORS.get(model_name, "blue"),
            alpha=0.7,
            label=model_name,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall (bubble size = mAP@0.5)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_map.png"), dpi=300)
    print("[*] Created precision-recall scatter plot")

    # ----- Radar Chart -----
    plt.figure(figsize=(10, 10))

    categories = ["Precision", "Recall", "F1-score", "mAP50", "mAP50-95"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)

    for i, row in results_df.iterrows():
        model_name = row["model"]
        values = [
            row["precision"],
            row["recall"],
            row["f1_score"],
            row["mAP50"],
            row["mAP50-95"],
        ]
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=model_name,
            color=COLORS.get(model_name, "blue"),
        )
        ax.fill(angles, values, alpha=0.1, color=COLORS.get(model_name, "blue"))

    plt.xticks(angles[:-1], categories)

    plt.ylim(0, 1)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "radar_comparison.png"), dpi=300)
    print("[*] Created radar comparison plot")

    print(f"[✓] All plots saved to {PLOTS_DIR}")


# ===== Main Evaluation Function =====
def main():
    print("\n" + "=" * 60)
    print("YOLO-TinyPerson Model Evaluation")
    print("=" * 60 + "\n")

    # ----- Check Hardware -----
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[i] Running evaluations on {device}")

    # ----- Find Valid Models -----
    valid_models = []
    for model in MODELS:
        if os.path.exists(model["model_path"]):
            model["size_mb"] = measure_model_size(model["model_path"])
            valid_models.append(model)
            print(f"[✓] Found {model['name']} model: {model['size_mb']:.2f} MB")
        else:
            print(f"[!] Model not found: {model['name']} at {model['model_path']}")

    if not valid_models:
        print("[ERROR] No valid models found. Please train models first.")
        return

    # ----- Evaluate All Models -----
    all_results = []
    for model in valid_models:
        results = evaluate_model(model)
        results["model_size_mb"] = model["size_mb"]
        all_results.append(results)

    # ----- Save Results to CSV -----
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[✓] Evaluation results saved to {csv_path}")

    # ----- Display Results Summary -----
    print("\n[i] Model Performance Summary:")
    print("-" * 100)
    print(
        f"{'Model':<10} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | "
        f"{'mAP50':>9} | {'mAP50-95':>9} | {'IoU':>9} | {'Speed (FPS)':>10} | {'Size (MB)':>9}"
    )
    print("-" * 100)

    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<10} | {row['precision']:>9.4f} | {row['recall']:>9.4f} | "
            f"{row['f1_score']:>9.4f} | {row['mAP50']:>9.4f} | {row['mAP50-95']:>9.4f} | "
            f"{row['mean_IoU']:>9.4f} | {row['inference_fps']:>10.2f} | {row['model_size_mb']:>9.2f}"
        )
    print("-" * 100)

    # ----- Generate Visualization Plots -----
    generate_plots(results_df)

    # ----- Completion Message -----
    print("\n" + "=" * 60)
    print(f"[✓] Evaluation complete! Results saved to {EVAL_DIR}")
    print("=" * 60)


# ===== Script Entry Point =====
if __name__ == "__main__":
    main()
