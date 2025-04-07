import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ===== Configuration =====
RESULTS_DIR = "results"
MODELS = ["YOLOv8x", "YOLOv9e", "YOLOv10x", "YOLO11x", "YOLO12x"]

PLOTS_DIR = "visualizations"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----- Model Color Scheme -----
COLORS = {
    "YOLOv8x": "#1f77b4",
    "YOLOv9e": "#ff7f0e",
    "YOLOv10x": "#2ca02c",
    "YOLO11x": "#d62728",
    "YOLO12x": "#9467bd",
}


# ===== Main Visualization Function =====
def main():
    print("\n" + "=" * 60)
    print("YOLO-TinyPerson Visualization")
    print("=" * 60 + "\n")

    # ----- Load Training Results -----
    print("[*] Loading training data for YOLO models...")

    dataframes = {}
    for model in MODELS:
        csv_path = os.path.join(RESULTS_DIR, model, "results.csv")
        if os.path.exists(csv_path):
            dataframes[model] = pd.read_csv(csv_path)
            print(f"[✓] Loaded data for {model}, {len(dataframes[model])} epochs")
        else:
            print(f"[!] Warning: No results file found for {model}")

    if not dataframes:
        print("[ERROR] No data found. Exiting.")
        exit()

    # ----- Configure Plot Style -----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["figure.figsize"] = (12, 8)

    print("\n[*] Generating comparative plots...")

    # ----- Training Loss Plot -----
    print("[*] Creating Training Loss plot")
    plt.figure()
    for model, df in dataframes.items():
        total_loss = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
        plt.plot(df["epoch"], total_loss, label=model, color=COLORS[model], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Combined Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_loss_comparison.png"), dpi=300)

    # ----- mAP@0.5 Plot -----
    print("[*] Creating mAP@0.5 plot")
    plt.figure()
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/mAP50(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5")
    plt.title("mAP@0.5 Comparison")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max([df["metrics/mAP50(B)"].max() for df in dataframes.values()]) * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "map50_comparison.png"), dpi=300)

    # ----- Precision Plot -----
    print("[*] Creating Precision plot")
    plt.figure()
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/precision(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision Comparison")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_comparison.png"), dpi=300)

    # ----- Recall Plot -----
    print("[*] Creating Recall plot")
    plt.figure()
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/recall(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall Comparison")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "recall_comparison.png"), dpi=300)

    # ----- Learning Rate Plot -----
    print("[*] Creating Learning Rate plot")
    plt.figure()
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"], df["lr/pg0"], label=model, color=COLORS[model], linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_rate_comparison.png"), dpi=300)

    # ----- Loss Component Plots -----
    print("[*] Creating individual loss component plots")
    loss_components = ["box_loss", "cls_loss", "dfl_loss"]

    for component in loss_components:
        plt.figure()
        for model, df in dataframes.items():
            plt.plot(
                df["epoch"],
                df[f"train/{component}"],
                label=model,
                color=COLORS[model],
                linewidth=2,
            )

        plt.xlabel("Epoch")
        plt.ylabel(f"{component.replace('_', ' ').title()}")
        plt.title(f"{component.replace('_', ' ').title()} Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{component}_comparison.png"), dpi=300)

    # ----- Combined Metrics Plot -----
    print("[*] Creating combined metrics plot")
    plt.figure(figsize=(14, 10))

    # Precision subplot
    plt.subplot(3, 1, 1)
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/precision(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )
    plt.ylabel("Precision")
    plt.title("Model Performance Metrics Comparison")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)

    # Recall subplot
    plt.subplot(3, 1, 2)
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/recall(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )
    plt.ylabel("Recall")
    plt.grid(True)
    plt.ylim(0, 1)

    # mAP subplot
    plt.subplot(3, 1, 3)
    for model, df in dataframes.items():
        plt.plot(
            df["epoch"],
            df["metrics/mAP50(B)"],
            label=model,
            color=COLORS[model],
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5")
    plt.grid(True)
    plt.ylim(0, max([df["metrics/mAP50(B)"].max() for df in dataframes.values()]) * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "combined_metrics_comparison.png"), dpi=300)

    # ----- Completion Message -----
    print(f"\n[✓] All plots saved to {Path(PLOTS_DIR).absolute()}")
    print("=" * 60)


# ===== Script Entry Point =====
if __name__ == "__main__":
    main()
