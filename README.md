
# AlphaNetworks

[![PyPI Version](https://img.shields.io/pypi/v/alphanetworks.svg)](https://pypi.org/project/alphanetworks/)
[![Python Versions](https://img.shields.io/pypi/pyversions/alphanetworks.svg)](https://pypi.org/project/alphanetworks/)
[![License](https://img.shields.io/pypi/l/alphanetworks.svg)](https://github.com/alphanetworks/alphanet/blob/main/LICENSE)

**AlphaNetworks** is a Python package designed to train advanced image classification models using hybrid architectures like **ResNet50V2** and **DenseNet169**. It provides a seamless interface for training, evaluating, and deploying deep learning models.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation via Pip](#installation-via-pip)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
    - [Basic Usage](#basic-usage)
    - [Options](#options)
    - [Help](#help)
  - [Examples](#examples)
  - [Programmatic Usage](#programmatic-usage)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
  - [Reporting Issues](#reporting-issues)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Introduction

**AlphaNetworks** combines state-of-the-art architectures to offer superior performance in image classification tasks. By leveraging pre-trained weights and advanced optimization techniques, it ensures robust feature extraction and better generalization.

---

## Features

- **Hybrid Architecture**: Integrates ResNet50V2 and DenseNet169.
- **Data Augmentation**: Enhances model robustness.
- **CLI Support**: Train and configure models via the command line.
- **Customizable Hyperparameters**: Adjust learning rate, batch size, etc.
- **Download Pretrained Weights**: Automatically fetch required weights.

---

## Installation

### Prerequisites

- **Python**: Version 3.6 or higher.
- **TensorFlow**: Version 2.x or later.
- **pip**: Latest version recommended.

### Installation via Pip

Install **AlphaNetworks** directly from PyPI:

```bash
pip install alphanetworks
```

---

## Usage

### Command-Line Interface

#### Basic Usage

```bash
alphanetworks --train TRAIN --val VAL [OPTIONS]
```

#### Options

- `--train` or `-t`: Path to the training dataset.
- `--val` or `-v`: Path to the validation dataset.
- `--epochs` or `-e`: Number of training epochs (default: 30).
- `--batch_size` or `-b`: Batch size for training and validation (default: 32).
- `--lr` or `-l`: Initial learning rate for the Adam optimizer (default: 0.001).
- `--output_dir` or `-o`: Directory to save model weights and reports.
- `--nc`: Number of target classes for classification (default: inferred from data).

#### Help

For a detailed list of options, run:

```bash
alphanetworks --help
```

---

### Examples

#### Example 1: Basic Training

```bash
alphanetworks --train ./data/train --val ./data/val --epochs 20
```

#### Example 2: Custom Parameters

```bash
alphanetworks --train ./data/train --val ./data/val --epochs 50 --batch_size 64 --lr 0.0005 --nc 10
```

---

### Programmatic Usage

```python
from alphanetworks import alphanet

# Build the model
model = alphanet(input_shape=(224, 224, 3), num_classes=10)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_data, validation_data=val_data, epochs=30)
```

---

## Troubleshooting

### Download Errors

During runtime, pre-trained weights for ResNet50V2 or DenseNet169 are automatically downloaded. Ensure you have a stable internet connection.

---

## Project Structure

```
alphanetworks/
├── alphanet.py
├── utils.py
├── scripts/
│   └── train.py
├── setup.py
└── README.md
```

---

## Documentation

For detailed documentation, visit the [GitHub repository](https://github.com/alphanetworks/alphanet).

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For any queries, reach out to the author at [mail](mailto:ihteshamjahangir21@gmail.com).

---

## Acknowledgments

- TensorFlow and Keras Teams
- Researchers of ResNet and DenseNet
