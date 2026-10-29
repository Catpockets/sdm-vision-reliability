# SDM vision reliability research write-up draft

**Preliminary clean-data analysis — October 20, 2026.** This draft covers five
completed classifiers and their available clean evaluations. EMNIST Balanced,
CIFAR-10, in-distribution SVHN and full corruption/OOD evaluation were unfinished
at the evidence cutoff. The results below are fixed to the accompanying
[aggregate evidence snapshot](assets/clean-results-2026-10-20.json), not a live
view of the campaign. The [analysis plan](experiment-design.md) governs the final
comparison and remaining figures.

## Abstract

We investigate whether similarity-distance-magnitude (SDM) confidence can support
useful selective prediction in multi-class vision. We compare a from-scratch
ResNet-18-style CIFAR-100 model with a pretrained ConvNeXt-Tiny family, evaluating
methods within each fixed checkpoint and label space. Calibration data fits
positive-temperature baselines and SDM estimators; evaluation reports probability
quality, error ranking, coverage and accuracy among admitted predictions.
Preliminary clean results show that temperature scaling reduces negative
log-likelihood and expected calibration error without changing top-1 predictions.
For CIFAR-100 ConvNeXt, larger SDM calibration support improves risk ranking but
neither calibration budget yields a qualifying 95% reliability region. The same
identity-adaptor implementation admits high-accuracy subsets on MNIST and KMNIST,
while Fashion-MNIST also yields no qualifying region. These observations motivate
separate analysis of ranking, calibration and admission feasibility. They do not
yet establish behavior under corruption or far-OOD inputs.

## 1. Introduction

A classifier's ability to predict a label and its ability to identify trustworthy
predictions are different evaluation questions. A model can be accurate overall
while producing a subset of confident errors. Selective prediction makes that
tradeoff explicit: the model may answer some inputs and abstain on others. Its
utility depends on both accuracy among accepted predictions and the fraction of
inputs it accepts. A method that accepts nothing supplies no useful classification
coverage, even if it avoids confident errors.

Our research question is whether SDM confidence improves this reliability/coverage
tradeoff relative to raw softmax confidence and temperature scaling on the same
vision model. SDM incorporates representation-based similarity and distance
signals in addition to decision scores. We test two implementations separately:
a learned adaptor applied to ResNet representations and the identity-adaptor
formulation in the ConvNeXt notebooks. They are not interchangeable estimator
variants or a controlled adaptor ablation.

The evaluation extends from clean classification to a planned fixed set of
corruptions and a separate far-OOD dataset. We use 95% accuracy on admitted
predictions as an empirical target. We do not assume that meeting that target on
calibration data guarantees it on a changed input distribution. Published-method
background and the project's reading references are maintained in the
[SDM reference](allen-schmaltz-sdm.md) and [upstream implementation notes](upstream-sdm.md);
a complete related-work synthesis remains outside these preliminary results.

## 2. Methods

### 2.1 Classifiers and data separation

We retain the existing source-notebook recipes. The ResNet-18-style model trains
from scratch on 32-pixel CIFAR-100 inputs for 30 epochs with backbone seed 42 and
batch size 128. It uses SGD with momentum 0.9, Nesterov updates, weight decay
0.0005 and a one-cycle schedule with maximum learning rate 0.1. The final epoch
is the evaluation checkpoint; validation does not select it. Its 512-dimensional
representations support the learned SDM fits.

ConvNeXt-Tiny starts from torchvision's fixed `IMAGENET1K_V1` weights. Its full
runs use seed 42, 20 epochs, batch size 128, AdamW at learning rate 0.0003,
weight decay 0.05, cosine scheduling and cross-entropy label smoothing 0.1.
Validation top-1 selects the checkpoint. CIFAR training uses stochastic crops,
horizontal flips, RandAugment and random erasing; evaluation resizes to 256 and
center-crops to 224 with ImageNet normalization. Additional datasets use their
own documented transforms, including RGB replication for grayscale images and
no digit/handwriting flips. These preprocessing and training differences prevent
attributing cross-backbone differences solely to uncertainty estimation.

| Experiment | Training/reference | Validation | Standard calibration | Enlarged calibration | Current clean evaluation |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet CIFAR-100 | 45,000 | 2,500 | 2,500 | Not run | 2,500 validation images |
| ConvNeXt CIFAR-100 | 37,500 | 2,500 | 2,500 | 10,000 | 10,000 official test images |
| Each completed extra dataset | 58,750 | 250 | 250 | 1,000 | 10,000 official test images |

Within each class, seed-42 permutations allocate disjoint validation and
calibration sets. The ConvNeXt standard calibration set is nested within its
enlarged set. Additional dataset notebooks reserve 25 validation and 100 enlarged
calibration examples per class, retaining all remaining training support; counts
in the last row apply specifically to MNIST, Fashion-MNIST and KMNIST. The two
backbones have different training support and clean evaluation populations, so
all method comparisons below remain within their own experiment blocks.

### 2.2 Confidence methods and admission rules

Raw softmax uses the maximum predicted class probability as confidence. Positive
temperature scaling divides logits by a scalar fitted using calibration NLL.
Temperature fitting does not use evaluation labels and preserves each image's
predicted fine class. ConvNeXt fits a separate temperature for each calibration
budget, using the same frozen backbone.

ConvNeXt uses the notebook-04 frozen identity-adaptor SDM implementation and
constructs its high-reliability region from each calibration set at alpha .95.
ResNet uses saved `reexpress-sdm` learned-adaptor artifacts, fitted with SDM seed 0
and 20 epochs for fine-100 and coarse-20 label spaces. The adaptor may change
predictions. The coarse task is analyzed separately; it is not evidence of an
improvement on the original 100-class problem. The completed learned-adaptor fits
are not yet included in the aggregate ResNet comparison below because the
matched evaluation has not finished.

The ResNet baseline results include its plain calibration-frozen temperature
threshold. Global softmax/temperature admission comparisons for ConvNeXt will be
supplied by the matched evaluator; their missing cells are not zero coverage.
That evaluator chooses attainable whole-confidence-group thresholds on calibration
only, with minimum calibration admissions of 1 for ResNet and 100 for ConvNeXt.
It reuses saved learned SDM policies for ResNet and the identity-adaptor region
for ConvNeXt. Its centroid/lower ResNet policies share central SDM probabilities;
they are separate admission policies, not independent models.

### 2.3 Metrics and uncertainty

We report full-coverage top-1 accuracy, NLL, multiclass Brier score, and expected
calibration error (ECE) using 15 bins. Smaller values of NLL, Brier and ECE are
better under their respective definitions; improvement in one need not imply
improvement in every other metric. Error AUROC/AUPRC treats prediction error as
the positive class. Risk-coverage calculations retain complete equal-confidence
groups, and AURC is the right-endpoint step integral including the first interval.
Lower AURC denotes less accumulated selective risk along that empirical ranking.
Retrospective curves do not fit the reported admission thresholds.

At each frozen admission rule, coverage is admitted count divided by evaluation
count; accepted accuracy is correct admitted count divided by admitted count.
Accepted accuracy is undefined with no admissions. We calculate 95% Wilson
intervals from saved per-image correctness and admission masks. These intervals
condition on the fitted rule and do not capture training-seed variability or
provide guarantees under shift. All results in this snapshot use one backbone
seed; no significance test for a method difference is claimed.

### 2.4 Remaining shift evaluation

The full matched protocol evaluates 15 standard CIFAR-100-C corruptions at five
severities and SVHN test images as far-OOD for each CIFAR model. Checkpoints,
representations, calibration identities and rules must be verified before those
results enter the write-up. CIFAR-100-C derives from official test images, so its
slices reuse base image identities and are not independent replicated trials.
SVHN far-OOD is evaluated using false-admission rates and ID-versus-OOD AUROC,
not CIFAR classification accuracy. ResNet validation and ConvNeXt official test
are their respective ID references. No shifted result or conclusion is supplied
in this draft.

## 3. Preliminary clean results

### 3.1 Temperature scaling improves ECE and NLL without changing accuracy

Table 1 separates ResNet validation from ConvNeXt official test. Accuracy and ECE
are percentages; NLL, Brier and AURC use their native scales. Calibration budgets
are examples per class. Raw softmax appears once per checkpoint even though the
collector repeats it in both paired ConvNeXt track groups.

**Table 1. CIFAR-100 clean probability and ranking metrics.**

| Population | Method | Calibration/class | Accuracy % | NLL | Brier | ECE % | AURC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet validation | Raw softmax | — | 73.84 | 1.0005 | 0.3699 | 8.61 | 0.07708 |
| ResNet validation | Temperature | 25 | 73.84 | 0.9394 | 0.3586 | 2.15 | 0.07762 |
| ConvNeXt test | Raw softmax | — | 88.45 | 0.5197 | 0.1790 | 5.96 | 0.03618 |
| ConvNeXt test | Temperature | 25 | 88.45 | 0.4945 | 0.1816 | 3.36 | 0.03394 |
| ConvNeXt test | Temperature | 100 | 88.45 | 0.4945 | 0.1815 | 3.45 | 0.03401 |
| ConvNeXt test | SDM identity | 25 | 88.45 | 2.2341 | 0.5416 | 47.00 | 0.04768 |
| ConvNeXt test | SDM identity | 100 | 88.45 | 1.7137 | 0.4251 | 34.34 | 0.02629 |

On ResNet validation, temperature scaling reduces ECE from 8.61% to 2.15% and
NLL from 1.0005 to 0.9394 while accuracy remains 73.84%. Its plain frozen rule
accepts 1,314 of 2,500 images (52.56% coverage); 1,243 of those predictions are
correct, giving 94.60% accepted accuracy with a 95% Wilson interval of
[93.24%, 95.69%]. The point estimate falls below the 95% target; an interval
containing 95% is not evidence that the target is maintained.

For ConvNeXt, temperature scaling reduces test ECE from 5.96% to 3.36% with
25 calibration examples per class, and to 3.45% with 100 per class. NLL decreases
from 0.5197 to approximately 0.4945 in both cases. Brier score increases slightly,
from 0.1790 to 0.1816/0.1815. Thus, the probability-metric changes are mixed rather
than a uniform improvement in every measure. Top-1 accuracy stays at 88.45%.

### 3.2 Better SDM ranking does not establish a usable reliability region

For CIFAR-100, enlarging SDM calibration support changes error AUROC from 0.8233
to 0.8915 and AURC from 0.04768 to 0.02629. The latter is below the raw-softmax
AURC of 0.03618. However, neither calibration budget yields a finite qualifying
region at alpha .95, and both frozen SDM policies admit zero test examples.
Accepted accuracy is therefore undefined. The retrospective ranking improvement
cannot be substituted for useful coverage at the prescribed admission rule.

The enlarged SDM track does not improve every confidence measure: its NLL is
1.7137 versus raw softmax’s 0.5197, and ECE is 34.34% versus 5.96%. Error AUPRC
is 0.4322 versus 0.5071 for raw softmax, despite the higher error AUROC. Thus,
there is no uniform ordering of the methods across calibration and ranking metrics.

The saved calibration summaries show distance signal zero fractions of 37.44%
and 14.61% in the standard and enlarged CIFAR tracks. These diagnostics help
identify questions about feature geometry and region construction; they do not
establish the cause of region failure. No post-test lowering of alpha or
threshold adjustment is used to turn the no-region outcome into a success.

### 3.3 Dataset checks show selective admissions on some clean tasks

**Table 2. Completed extra-dataset classifier accuracy and standard-track
softmax-to-temperature comparison.** Both methods share the listed test accuracy;
NLL uses its native scale and ECE is a percentage.

| Dataset | Test accuracy % | NLL raw → temperature | ECE % raw → temperature |
| --- | ---: | ---: | ---: |
| mnist | 99.65 | 0.1030 → 0.0190 | 8.72 → 0.18 |
| fashion_mnist | 95.02 | 0.2244 → 0.1853 | 6.85 → 1.74 |
| kmnist | 98.45 | 0.1386 → 0.0740 | 8.17 → 0.52 |

These three completed datasets show lower temperature-scaled ECE and NLL, but
similar-looking classification accuracies do not imply the same SDM admission
behavior. Table 3 uses each dataset's saved identity-adaptor policy at alpha .95.
Its intervals are calculated from the saved per-image decisions, rather than
filled from missing aggregate interval columns.

**Table 3. Frozen SDM admissions on completed ConvNeXt clean tests.** Accuracy
and coverage are percentages. Each bracket is a 95% Wilson interval conditional
on the fitted rule; 100% observed accuracy is not certainty of future correctness.

| Dataset | Calibration/class | Region found on calibration | Admitted / total | Coverage % | Accepted accuracy % [95% interval] |
| --- | ---: | --- | ---: | ---: | --- |
| cifar100 | 25 | No | 0 / 10,000 | 0.00 | Undefined |
| cifar100 | 100 | No | 0 / 10,000 | 0.00 | Undefined |
| mnist | 25 | Yes | 6,069 / 10,000 | 60.69 | 100.00 [99.94, 100.00] |
| mnist | 100 | Yes | 4,981 / 10,000 | 49.81 | 100.00 [99.92, 100.00] |
| fashion_mnist | 25 | No | 0 / 10,000 | 0.00 | Undefined |
| fashion_mnist | 100 | No | 0 / 10,000 | 0.00 | Undefined |
| kmnist | 25 | Yes | 3,314 / 10,000 | 33.14 | 100.00 [99.88, 100.00] |
| kmnist | 100 | Yes | 3,992 / 10,000 | 39.92 | 99.95 [99.82, 99.99] |

MNIST admits 60.69% of test images in the standard track and 49.81% in the
enlarged track, with no observed admitted errors in either. More calibration data
does not monotonically increase coverage. KMNIST admits 33.14% and 39.92%, with
zero and two admitted errors respectively. Fashion-MNIST has no qualifying region
in either track despite 95.02% full-coverage accuracy. These observations do not
support a universal claim that SDM forms a useful 95% region for vision tasks.
They also do not establish an advantage over global softmax/temperature admission
rules; that paired frozen-rule comparison remains pending matched evaluation.

## 4. Failure analysis and limitations

Three distinct behaviors must remain separate: the model's label accuracy,
probability calibration, and the usefulness of its admission rule. CIFAR-100's
enlarged SDM track improves AURC while failing to form a qualifying region; MNIST's
larger calibration track reduces coverage despite perfect observed admitted
accuracy. These are findings to explain, not reasons to select a favorable metric
and suppress the rest.

Possible explanations include class-conditional support, tied confidence groups,
distance/similarity distributions and differences in region construction. No
controlled ablation in this snapshot identifies one of these as the mechanism.
The learned ResNet adaptor and ConvNeXt identity adaptor also differ in predictions
and fitting, so their behavior cannot isolate an architecture effect.

There is one backbone seed per dataset, and the four completed ConvNeXt tasks are
only a subset of the planned seven. The protocols use different architectures,
pretraining, training supports, transforms and checkpoint-selection rules.
ResNet validation is not the ConvNeXt final-test population. Historical project
test exploration and preparation of the analysis plan after viewing some results
also prevent presenting this work as a fully blind preregistered evaluation.
CIFAR/SVHN experiments do not establish deployed inspection or safety performance.

## 5. Questions for the SDM methodology discussion

Ask the method author to distinguish estimator assumptions from implementation
choices before adding new experiments:

1. In the identity-adaptor implementation, which calibration diagnostics best
   explain no finite alpha-.95 region despite improved error ranking? Supply the
   observed CIFAR distance-zero fractions and class-support summaries.
2. Are the learned-adaptor and identity-adaptor results answering comparable SDM
   questions, or should they be presented as separate transfer experiments?
3. How should a global empirical-accuracy threshold be compared with the upstream
   SDM high-reliability policy when their support rules and statistical criteria
   differ? What matched sensitivity analysis would be appropriate?
4. How should the report interpret smaller coverage under a larger calibration
   budget, and what evidence would distinguish expected conservatism from a
   configuration problem?

These are discussion questions based on completed clean outputs, not claims that
a meeting occurred or authorization to change the running campaign.

## 6. Remaining results and final conclusion

EMNIST Balanced, CIFAR-10 and in-distribution SVHN results must be added after
completion checks. Full corruption/OOD runs must supply frozen-rule reliability,
coverage, confidently wrong rates and separate far-OOD admissions. The remaining
T1–T5/F1–F5 displays and results gate are specified in the
[analysis plan](experiment-design.md#results-tables-and-figure-plan).

The current evidence supports a narrow conclusion: temperature scaling lowers ECE
and NLL on the completed clean tasks; SDM's ranking and admission-region outcomes
are mixed and must be reported separately. Whether these methods preserve useful
reliability under distribution shift remains unanswered by this snapshot.

## Reproducibility appendix

The [evidence snapshot](assets/clean-results-2026-10-20.json) records the UTC cutoff,
completed/pending inventory, exact checkpoint hashes, metric-definition version,
calibration tracks, aggregate values, per-image-derived admission counts/intervals,
and hashes of the supporting artifacts. Source files are identified relative to
the campaign root so the evidence can be audited without committing raw predictions,
images or checkpoints. New full-run outputs must not silently change this dated
snapshot; update the report deliberately when the remaining evidence is verified.

The scientific campaign is bound to source revision
`782d99ac9e17f38b06e256ced6f58dda2006d0fc`. Later checkpoint-runtime fixes in the
review branches do not retroactively alter that run. Raw/temperature accuracy,
NLL, Brier and ECE were recomputed from the saved evaluation outputs and fitted
temperatures. Native SDM metric rows were reconciled to their exported tables;
SDM test identities, admission counts and Wilson intervals were checked against
per-image masks. This is evidence reconciliation, not an independent replication
of the full SDM fit or a significance test between methods.

For a reproduced campaign directory, collect completed benchmark evidence with:

```bash
python scripts/collect_experiment_results.py --campaign /path/to/campaign \
  --output-dir /path/to/results
```

When matched runs finish, add their two directories using `--shift-dirs` as shown
in the [matched-evaluation guide](matched-shift-evaluation.md). Keep comparison
groups and evaluation populations separate before updating Tables 1–3 or adding
new conclusions. The manuscript's rounded table cells are derived from the
snapshot's full-precision values; percentages multiply stored rates by 100.
