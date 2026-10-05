# SDM Vision Reliability

Building reliable vision models with **Similarity-Distance-Magnitude (SDM)** uncertainty estimation, selective prediction, and out-of-distribution detection.

## Overview

Modern deep learning models can achieve strong benchmark accuracy while still making highly confident predictions on corrupted, ambiguous, or out-of-distribution inputs.

This project investigates whether the **Similarity-Distance-Magnitude (SDM)** framework can improve reliability in multi-class computer vision systems by identifying predictions that should not be trusted.

The goal is to build models that can selectively abstain from uncertain predictions while maintaining high accuracy on the predictions they do accept.

## Project Documents

- [Start here: CIFAR-100 EDA notebook](notebooks/01_cifar100_eda.ipynb)
- [Train a ResNet-18 CNN on CIFAR-100](notebooks/02_cifar100_cnn_resnet18.ipynb)
- [Experiment design: from-scratch ResNet-18 as the primary model](docs/experiment-design.md)
- [Initial project idea and requirements](docs/project-brief.md)
- [Allen Schmaltz’s research overview and attached poster](docs/allen-schmaltz-sdm.md)
- [Pinned upstream SDM checkout](docs/upstream-sdm.md)
- [Upstream SDM research code](https://github.com/ReexpressAI/sdm_activations)
- [Project issues and starter tasks](https://github.com/Catpockets/sdm-vision-reliability/issues)

## Research Question

Can SDM-based uncertainty estimation improve the reliability of multi-class vision models under distribution shift and out-of-distribution conditions?

## Project Goals

* Apply SDM uncertainty estimation to multi-class image classification
* Evaluate model confidence under corrupted and out-of-distribution inputs
* Compare SDM against standard confidence and calibration methods
* Measure the tradeoff between prediction coverage and reliability
* Identify high-reliability prediction regions
* Explore extensions to multimodal vision-language models

## Datasets

Initial experiments will focus on:

* **CIFAR-100** — primary multi-class classification benchmark
* **CIFAR-100-C** — corrupted-image robustness benchmark
* **SVHN** — out-of-distribution evaluation dataset

Potential future work may include vision-language model hallucination benchmarks such as **POPE**.

## Methods

The project may evaluate techniques including:

* Similarity-Distance-Magnitude (SDM)
* Selective prediction
* Out-of-distribution detection
* Confidence calibration
* Temperature scaling
* Conformal prediction
* Deep feature-space analysis
* Vision Transformers
* Convolutional neural networks

## Evaluation

Key metrics may include:

* Classification accuracy
* Conditional accuracy
* Coverage
* Risk-coverage curves
* Expected calibration error
* Out-of-distribution detection performance
* Abstention rate
* False-confidence / catastrophic-error rate

A primary objective is to determine whether a model can maintain a target reliability level, such as **95% accuracy on admitted predictions**, while rejecting uncertain inputs. This is a research target, not an established guarantee for arbitrary distribution shifts. Experiments must report coverage alongside accuracy, uncertainty in the estimates, and the assumptions behind any claimed guarantee.

## Tech Stack

The primary language for this project is **Python**.

Expected tools and libraries include:

```text
Python
PyTorch
torchvision
NumPy
pandas
scikit-learn
Matplotlib
Jupyter
```

Additional libraries may be introduced as the project evolves.

## Repository Structure

Start with the notebook to get familiar with the data:

```text
sdm-vision-reliability/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_cifar100_eda.ipynb
│   └── 02_cifar100_cnn_resnet18.ipynb
└── docs/
    ├── experiment-design.md
    ├── project-brief.md
    ├── allen-schmaltz-sdm.md
    └── assets/
        └── poster.png
```

The notebook explores CIFAR-100 training images, class names, class balance, and
pixel values. It includes questions for the team to discuss before modeling.

The ResNet-18 notebook trains a CNN from scratch with PyTorch, using the same
training/validation/calibration split as the ConvNeXt baseline. It fits temperature
scaling and a 95%-selective-accuracy abstention threshold on the calibration split and
checks both on validation, as the baseline for SDM. It saves the model and its
validation/calibration logits and embeddings to `outputs/` (ignored by Git). Training
takes about 30 minutes on an Apple-silicon or NVIDIA GPU.

## Getting Started

```bash
git clone https://github.com/Catpockets/sdm-vision-reliability.git
cd sdm-vision-reliability
```

### Open the first-look notebook

Use Python 3.12. From the repository folder, create an environment outside the
repository, install the dependencies, and start JupyterLab:

```bash
python3.12 -m venv ~/.venvs/sdm-eda
source ~/.venvs/sdm-eda/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/01_cifar100_eda.ipynb
```

On Windows PowerShell, use these first two commands instead:

```powershell
py -3.12 -m venv "$HOME\.venvs\sdm-eda"
& "$HOME\.venvs\sdm-eda\Scripts\Activate.ps1"
```

In JupyterLab, choose **Kernel → Restart Kernel and Run All Cells**. The first run
unpacks `data/cifar-100-python.tar.gz` if you have put the official archive there;
otherwise it downloads about 161 MB of CIFAR-100 data from a Kaggle mirror via
`kagglehub` (no Kaggle account needed) into `~/.cache/kagglehub/`. Later runs reuse
the files, and the notebook checks each one against the official checksums. No GPU
setup is required.

You will see image and class galleries, dataset dimensions, label counts, and
color histograms. Try changing `CLASS_NAME` and write your observations in the
last cell. This first notebook uses only training data; the test set and
corruption/OOD experiments are reserved for later work.

Keep downloaded data and local environments outside Git; the `.gitignore` already
excludes `data/`. The notebook is saved with cleared outputs; use **Run All** to populate
the tables and plots. Clear outputs before committing notebook edits.

For project context, read the [project brief](docs/project-brief.md) and
[research reference](docs/allen-schmaltz-sdm.md).

## Status

🚧 **Active Research Project**

The methodology, experiments, and repository structure will evolve throughout development.

## Team

UC Berkeley Master of Information and Data Science (MIDS) Capstone Project

Randell, Yiwen, and Karim. Original team annotations and presentation placeholders are in the [project brief](docs/project-brief.md).

## Acknowledgments

This project builds on research into the **Similarity-Distance-Magnitude (SDM)** framework and broader work in uncertainty estimation, selective prediction, model calibration, and out-of-distribution detection.
