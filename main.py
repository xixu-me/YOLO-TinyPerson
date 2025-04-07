import argparse
import os
import sys
import time
from pathlib import Path

# ===== Import Project Modules =====
import evaluate
import setup
import train
import visualize


# ===== Display Banner Function =====
def display_banner():
    """Display the project banner in the console."""
    print("\n" + "=" * 60)
    print("  YOLO-TinyPerson: Tiny Person Detection with YOLO Models")
    print("=" * 60)
    print("  A framework for training and evaluating YOLO models")
    print("  on the TinyPerson dataset for small object detection")
    print("=" * 60 + "\n")


# ===== Command Functions =====
def setup_environment(args):
    """Set up the environment, datasets, and models."""
    print("\n[*] Setting up environment...")
    setup.main()


def train_models(args):
    """Train YOLO models on the TinyPerson dataset."""
    print("\n[*] Training models...")
    train.main()


def evaluate_models(args):
    """Evaluate trained YOLO models on the TinyPerson dataset."""
    print("\n[*] Evaluating models...")
    evaluate.main()


def visualize_results(args):
    """Create visualizations of training and evaluation results."""
    print("\n[*] Generating visualizations...")
    visualize.main()


def run_all(args):
    """Run the complete pipeline: setup, train, evaluate, and visualize."""
    print("\n[*] Running complete pipeline...")

    start_time = time.time()

    print("\n[1/4] Setting up environment")
    setup.main()

    print("\n[2/4] Training models")
    train.main()

    print("\n[3/4] Evaluating models")
    evaluate.main()

    print("\n[4/4] Generating visualizations")
    visualize.main()

    # ----- Calculate Total Execution Time -----
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 60)
    print(
        f"[✓] Complete pipeline finished in {int(hours)}h {int(minutes)}m {int(seconds)}s"
    )
    print("=" * 60)


# ===== Command Line Interface =====
def create_parser():
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="YOLO-TinyPerson: Tiny Person Detection with YOLO Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ----- Setup Command -----
    setup_parser = subparsers.add_parser(
        "setup", help="Set up environment, download dataset and models"
    )
    setup_parser.set_defaults(func=setup_environment)

    # ----- Train Command -----
    train_parser = subparsers.add_parser(
        "train", help="Train YOLO models on TinyPerson dataset"
    )
    train_parser.set_defaults(func=train_models)

    # ----- Evaluate Command -----
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.set_defaults(func=evaluate_models)

    # ----- Visualize Command -----
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate visualizations of results"
    )
    viz_parser.set_defaults(func=visualize_results)

    # ----- All Command -----
    all_parser = subparsers.add_parser("all", help="Run complete pipeline")
    all_parser.set_defaults(func=run_all)

    return parser


# ===== Main Function =====
def main():
    """Main entry point for the application."""
    display_banner()

    parser = create_parser()
    args = parser.parse_args()

    # ----- Check for Command (use 'all' as default) -----
    if not args.command:
        print("[i] No command specified. Running complete pipeline by default.\n")
        run_all(args)
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


# ===== Script Entry Point =====
if __name__ == "__main__":
    main()
