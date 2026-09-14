# SDM Vision Reliability

Building reliable vision models with **Similarity-Distance-Magnitude (SDM)** uncertainty estimation, selective prediction, and out-of-distribution detection.

## Overview

Modern deep learning models can achieve strong benchmark accuracy while still making highly confident predictions on corrupted, ambiguous, or out-of-distribution inputs.

This project investigates whether the **Similarity-Distance-Magnitude (SDM)** framework can improve reliability in multi-class computer vision systems by identifying predictions that should not be trusted.

The goal is to build models that can selectively abstain from uncertain predictions while maintaining high accuracy on the predictions they do accept.

## Project Documents

- [Initial project idea and requirements](docs/project-brief.md)
- [Dataset preparation, splits, and loading checks](docs/datasets.md)
- [Allen Schmaltz’s research overview and attached poster](docs/allen-schmaltz-sdm.md)
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

The repository currently contains project documentation:

```text
sdm-vision-reliability/
├── README.md
└── docs/
    ├── project-brief.md
    ├── allen-schmaltz-sdm.md
    └── assets/
        └── poster.png
```

Dataset utilities, notebooks, model code, experiment configurations, tests, and dependencies will be added as implementation tasks are completed.

## Getting Started

```bash
git clone https://github.com/Catpockets/sdm-vision-reliability.git
cd sdm-vision-reliability
```

Read the [project brief](docs/project-brief.md) and [research reference](docs/allen-schmaltz-sdm.md), then choose a [starter issue](https://github.com/Catpockets/sdm-vision-reliability/issues). There is no runnable training pipeline or dependency manifest yet; environment setup is a starter task.

## Status

🚧 **Active Research Project**

The methodology, experiments, and repository structure will evolve throughout development.

## Team

UC Berkeley Master of Information and Data Science (MIDS) Capstone Project

Randell, Yiwen, and Karim. Original team annotations and presentation placeholders are in the [project brief](docs/project-brief.md).

## Acknowledgments

This project builds on research into the **Similarity-Distance-Magnitude (SDM)** framework and broader work in uncertainty estimation, selective prediction, model calibration, and out-of-distribution detection.
