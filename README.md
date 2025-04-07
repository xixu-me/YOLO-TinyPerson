# YOLO-TinyPerson

A comprehensive framework for training and evaluating YOLO models on the TinyPerson dataset for small object detection. This repository provides tools to benchmark different YOLO model versions (v8, v9, v10, v11, v12) on detecting small persons in images.

## Features

- **Complete Pipeline**: Setup environment, train models, evaluate performance, and visualize results
- **Multiple YOLO Models**: Support for YOLOv8x, YOLOv9e, YOLOv10x, YOLO11x, and YOLO12x
- **Automated Setup**: Downloads required datasets, models, and dependencies
- **Comprehensive Evaluation**: Measures precision, recall, F1 score, mAP, and inference speed
- **Rich Visualizations**: Generates performance comparison plots across models

## Prerequisites

- Python 3.6+
- CUDA-compatible GPU recommended (training on CPU is supported but not recommended)
- Windows or Linux operating system

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/YOLO-TinyPerson.git
   cd YOLO-TinyPerson
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set `dataset_dir` to the dataset path in [`setup.py`](setup.py):

   ```python
   dataset_dir = "path/to/dataset"
   ```

## Usage

The framework provides a simple command-line interface for running different parts of the pipeline:

### Complete Pipeline

Run the entire process (setup, train, evaluate, visualize):

```bash
python main.py all
```

or just:

```bash
python main.py
```

This will execute the default command, which is `all`.

### Individual Steps

Setup environment and download required files:

```bash
python main.py setup
```

Train YOLO models on the TinyPerson dataset:

```bash
python main.py train
```

Evaluate trained models:

```bash
python main.py evaluate
```

Generate visualizations of training and evaluation results:

```bash
python main.py visualize
```

## Project Structure

```
YOLO-TinyPerson/
├── main.py           # Main entry point with command-line interface
├── setup.py          # Environment setup, downloads datasets and models
├── train.py          # Model training functionality
├── evaluate.py       # Model evaluation and metrics calculation
├── visualize.py      # Results visualization and plot generation
├── requirements.txt  # Required Python packages
├── dataset/          # TinyPerson dataset (downloaded by setup.py)
├── weights/          # Pre-trained model weights (downloaded by setup.py)
├── results/          # Training results for each model
├── evaluation/       # Evaluation metrics and results
└── visualizations/   # Generated plots and visualizations
```

## Performance Metrics

The framework evaluates models using the following metrics:

- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all relevant instances
- **F1-score**: Harmonic mean of precision and recall
- **mAP@0.5**: Mean Average Precision with IoU threshold of 0.5
- **mAP@0.5-0.95**: Mean Average Precision across multiple IoU thresholds
- **Inference Speed**: Frames processed per second

Visualizations are generated in the `visualizations/` directory, including:

- Precision/Recall/F1 comparison across models
- mAP comparison
- Training loss curves
- Inference speed comparison
- Radar charts for overall performance comparison

## Model Support

The following YOLO models are supported:

| Model | Description |
|-------|-------------|
| YOLOv8x | YOLOv8 extra large model |
| YOLOv9e | YOLOv9 extended model |
| YOLOv10x | YOLOv10 extra large model |
| YOLO11x | YOLO11 extra large model |
| YOLO12x | YOLO12 extra large model |

## Acknowledgments

- The TinyPerson dataset is used for training and evaluation
- YOLO models from the Ultralytics framework

## License

Copyright &copy; [Xi Xu](https://xi-xu.me). All rights reserved.

Licensed under the [GPL-3.0](LICENSE) license.
