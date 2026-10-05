# Experiment design: from-scratch ResNet-18 as the primary model

This document proposes the main model and evaluation plan for the SDM experiments. It is a proposal for team discussion, not a record of results. Numbers marked *expected* are estimates to be replaced by measured values.

## Decision

Use a **ResNet-18 trained from scratch on CIFAR-100** ([notebook](../notebooks/02_cifar100_cnn_resnet18.ipynb)) as the primary model for calibration and SDM experiments. Keep the ImageNet-pretrained ConvNeXt-Tiny ([baseline notebook](../notebooks/02_cifar100_training_baseline.ipynb)) as an optional secondary comparison.

## Why not a pretrained model as the primary model

A pretrained model has learned from data we did not choose and cannot fully inspect. ConvNeXt-Tiny's `IMAGENET1K_V1` weights were trained on about 1.28 million ImageNet photos in 1,000 categories, and many CIFAR-100 categories also appear in ImageNet. That makes several conclusions harder to defend:

- **Out-of-distribution tests.** An input is only "unfamiliar" if the *model* has not seen anything like it. With pretraining, we cannot easily rule out exposure to similar images or categories.
- **Attribution.** If confidence or SDM behaves well, we cannot separate the effect of the method from the effect of the unknown pretraining data.
- **Reproducibility.** Every image a from-scratch model learned from is in our training split, with a fixed seed and documented augmentation.

## What we give up

- **Accuracy.** Expected clean top-1 is roughly 70–75% for the from-scratch ResNet-18, versus a ~90% target for the pretrained ConvNeXt. At a 95% selective-accuracy target, the ResNet will therefore answer fewer images.
- This does not undermine the research question, which is **relative**: on the same model, does SDM answer more images than softmax confidence or temperature scaling at the same accuracy, and does it reject corrupted and out-of-distribution inputs more reliably?

## Data splits

The split is identical to the ConvNeXt baseline notebook (within each class, shuffle with seed 42; first 25 images to validation, next 25 to calibration), so the two models can be compared on the same images.

| split | images | used for |
|---|---|---|
| training | 45,000 | learning network weights only |
| validation | 2,500 | monitoring training; checking fitted thresholds |
| calibration | 2,500 | fitting temperature scaling, SDM, and abstention thresholds |
| test (official) | 10,000 | final evaluation once, after all choices are frozen |

The ResNet-18 run keeps the **final-epoch** model, so validation is not used to select the checkpoint.

## Methods compared (all on the same trained model)

1. **Softmax confidence**: the raw maximum softmax probability.
2. **Temperature scaling**: one temperature fitted on the calibration split by minimizing negative log-likelihood.
3. **SDM**: similarity, distance, and magnitude signals computed from the model's 512-dimensional embeddings and logits, fitted on the calibration split.

Each method produces a confidence score and an abstention threshold chosen on the calibration split for a target selective accuracy of 95%. We report both a plain threshold and a conservative one (lower end of the 95% Wilson interval ≥ target).

## Evaluation conditions

| condition | data | question |
|---|---|---|
| clean | validation, then official test | Are accuracy and calibration good on ordinary inputs? |
| corrupted | CIFAR-100-C (15 standard corruptions, 5 severities) | Does the threshold still deliver its target accuracy as inputs degrade? |
| out-of-distribution | SVHN test set | Does the method reject images from a different domain? |

## Metrics

- Top-1 accuracy (top-5 for context)
- Expected calibration error (15 bins) and negative log-likelihood
- Coverage and selective accuracy at the chosen threshold, with 95% intervals
- Accuracy–coverage (risk–coverage) curves
- For SVHN: fraction of out-of-distribution images admitted, and AUROC for separating in- from out-of-distribution
- High-confidence error rate: wrong predictions that pass the threshold

## Uncertainty in results

- Train 3–5 seeds of the ResNet-18 and report mean and spread for every metric. Each run takes about 30 minutes on an Apple M3 Pro.
- Report interval estimates for selective accuracy; 2,500 validation images leave room for chance.
- A threshold that meets 95% on calibration data is not a guarantee on shifted data. Results under CIFAR-100-C and SVHN are measured, not assumed.

## Known limitations

- **Near-duplicates inside CIFAR-100.** Prior work reports that a noticeable share of CIFAR-100 test images are near-duplicates of training images (Barz & Denzler, "Do We Train on Test Data? Purging CIFAR of Near-Duplicates"; citation to verify). This affects every CIFAR-100 model, including from-scratch ones. A cleaned test set (ciFAIR-100) could be used as a robustness check.
- **One architecture.** Conclusions from one from-scratch CNN may not transfer to pretrained models or transformers. The optional ConvNeXt comparison addresses this partly.

## Open questions for the team

1. Do we agree on the from-scratch ResNet-18 as the primary model, with ConvNeXt as a secondary comparison?
2. How many seeds can we afford, and who runs them (local Mac vs. AWS credits)?
3. Should the final test evaluation use the official CIFAR-100 test set, ciFAIR-100, or both?
4. Is 95% the right selective-accuracy target, or should we report several targets (90%, 95%, 99%)?
