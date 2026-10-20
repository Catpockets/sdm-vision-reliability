# Experiment design, analysis plan and write-up structure

This is the team's working analysis plan, grounded in notebooks 02–10 and the
matched evaluator in PR #23. Karim requested both the from-scratch ResNet-18 and
pretrained ConvNeXt-Tiny tracks. Yiwen's proposal to foreground the from-scratch
model remains a presentation decision for the team; both tracks are retained.
This document replaces earlier assumptions that their training splits were equal
or that several backbone seeds had already been scheduled.

The analysis structure was prepared while the campaign was running, after the
ResNet validation results and historical project test results had been examined.
It is not a preregistration or evidence that the official test population has
never previously been explored. No unfinished benchmark result is asserted here.
Live results and run state belong in the exported artifacts, not this document.

## Research questions and comparison rules

1. Within a fixed checkpoint, label space, calibration track and evaluation
   population, does SDM offer a better empirical reliability/coverage tradeoff
   than raw softmax or temperature scaling?
2. Do calibration-frozen admission rules retain their clean behavior as corruption
   severity increases, and how often do they admit SVHN far-OOD images?
3. How do the nested calibration budgets affect ConvNeXt reliability? The two
   budgets reuse the same checkpoint and nested examples; they are paired tracks,
   not independent model runs.
4. Do the six additional in-distribution datasets show consistent or mixed
   patterns? These are generalization checks across datasets, not replacements
   for CIFAR corruption/OOD evaluation.

Compare methods within the collector's `comparison_group`, which binds dataset,
condition, class count, evaluation split, checkpoint hash, seed, calibration track
and metric definition. Do not pool ResNet with ConvNeXt, fine-100 with coarse-20,
validation with official test, or different checkpoints into one method ranking.
A cross-backbone overview is descriptive: architectures, pretraining, input
transforms, training support and checkpoint selection all differ.

## Executed recipes and evaluation populations

| Track | Training/reference | Validation | Calibration | Backbone/checkpoint | Clean reporting population |
| --- | --- | --- | --- | --- | --- |
| ResNet CIFAR-100 | 45,000 | 2,500 | 2,500 (25/class) | From scratch; seed 42; 30 epochs; batch 128; final epoch | Validation in notebooks 02/03 and current matched evaluator |
| ConvNeXt CIFAR-100 | 37,500 | 2,500 | Nested 2,500/10,000 (25/100 per class) | ImageNet `IMAGENET1K_V1`; seed 42; 20 epochs; batch 128; best validation epoch | Official 10,000-image test after current-run fits freeze |
| ConvNeXt six datasets | All official training examples remaining after reserving 125/class | 25/class | Nested 25/100 per class | Same pretrained family; seed 42; 20 epochs; batch 128; best validation epoch | Each dataset's own official test |

The six datasets are CIFAR-10, MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced
(47 classes) and SVHN. Preserve natural imbalance in the remaining training data.
Within-class seed-42 permutations define CIFAR split membership; the ResNet and
ConvNeXt training supports differ even where validation/standard-calibration IDs
match. Verify exported IDs rather than inferring identity from counts alone.

ResNet uses its own 32-pixel preprocessing; ConvNeXt uses its 224-pixel pretrained
pipeline. Details and versions come from the source notebooks and run manifests.
The ResNet SDM fits use seed 0 and 20 epochs, separately for fine-100 and coarse-20.
Only one backbone seed is in the current campaign. Additional seeds, new
architectures and new targets are further experiments requiring explicit recipes;
none are represented as completed or queued by this plan.

ResNet's clean official test remains unopened by this campaign's ResNet notebooks.
CIFAR-100-C uses official test-derived images/labels: its evaluation is therefore
held-out test-derived evidence even without a clean ResNet test pass. The current
matched evaluator reports ResNet validation versus ConvNeXt official test as their
respective clean/OOD references. Do not silently substitute one for the other.

## Methods and frozen operating points

- Raw softmax and temperature scaling are evaluated on the same backbone outputs.
  Fit temperature on calibration NLL only; never refit it for a corruption, severity
  or OOD input. Positive temperature preserves the image's predicted fine class.
- ConvNeXt SDM is the notebook-04 identity-adaptor formulation, fitted separately
  on each calibration budget. It does not constitute a learned-adaptor ablation.
- ResNet uses the saved learned SDM adaptor and separate fine/coarse artifacts;
  its predictions can differ from the backbone. Report these changes alongside
  confidence behavior. Coarse baselines sum fine-class probabilities into groups;
  their temperature is inherited from the fine-class fit, not newly optimized for
  coarse labels.
- ResNet SDM centroid/lower rows are two admission policies on one fitted model.
  Both use central SDM probabilities for probability and ranking metrics. Repeated
  NLL/ECE/AURC values across those policies are not independent findings.
- The matched softmax/temperature rules maximize attainable calibration coverage
  at empirical accepted accuracy >=95%, retaining complete equal-confidence
  groups. The current minimum admitted calibration count is 1 for ResNet and 100
  for ConvNeXt. SDM uses its upstream high-reliability policy at alpha .95. These
  are distinct admission procedures; report their names and support, not an
  implication that all methods share the same statistical bound.
- The original ResNet notebook additionally exports a Wilson-lower-bound baseline
  rule. The matched evaluator currently compares the plain empirical baseline
  rule. Do not describe its rows as Wilson-conservative baseline results.
- Freeze checkpoint, temperature, calibration choices and admission rules before
  current-run test evaluation. Never choose a new threshold or alpha from shifted
  accuracy. An absent qualifying region means zero admissions, with undefined
  accepted accuracy; lower-alpha regions are diagnostic evidence, not success at
  the .95 target.

## Analysis sequence and decision rules

First check completion metadata, checkpoint/source identities, split membership,
replay checks and class mapping. Include completed benchmark outputs only; retain
failed/queued/no-region outcomes in the inventory. Smoke checks establish wiring,
not research accuracy. Older outputs using the prior tied-score convention are
not interchangeable with `grouped-threshold-step-aurc-v1`.

For clean data, report full-coverage performance before selective results. At each
frozen rule report total and admitted counts, coverage, accepted accuracy and its
95% Wilson interval. The key reliability/utility pair is accepted accuracy **and**
coverage. A rule attaining the calibration target can miss it on evaluation; show
that miss directly. A point estimate >=.95 and an interval lower endpoint >=.95
are different statements. No-admission results cannot establish 95% reliability.

Then evaluate all 15 standard corruptions at severities 1–5 with those same frozen
rules. Show per-condition results, target shortfalls and coverage collapse rather
than reporting only favorable corruptions. A uniform average across the 75 cells
may be labelled a descriptive benchmark average; it is not a deployment mixture.
All severities/corruptions reuse base test images and are not 75 independent runs.

For SVHN far-OOD, report false-admission rate (lower is better), rejection rate
(1 minus false-admission), and ID-versus-OOD AUROC (ID is positive; higher is
better). State whether the ID reference is ResNet validation or ConvNeXt official
test. Digit labels are audit-only; CIFAR classification accuracy/NLL is undefined
on SVHN OOD. In-distribution SVHN notebook 10 is a separate classification task.

Finally compare ConvNeXt's calibration budgets and six dataset results without
selecting datasets by final-test accuracy. Separate probability calibration from
error ranking and from threshold admissions: improvement in one does not imply
improvement in the others. Include negative and mixed outcomes.

## Metrics, uncertainty and reporting conventions

| Quantity | Definition or convention | Evidence |
| --- | --- | --- |
| Coverage | admitted / all examples | `frozen_rule_coverage`; counts in matched metrics |
| Accepted accuracy | correct admitted / admitted; undefined with none | `frozen_rule_accuracy` and Wilson interval columns |
| Selective risk | 1 minus accepted accuracy | Same frozen-rule counts; risk curves separately |
| Confidently wrong rate | wrong admitted / all examples, not / admitted | Derive from saved labels/predictions/admission masks |
| Classification | Full-coverage top-1; macro/micro and per-class precision/recall | Aggregate table plus native per-class exports |
| Probability calibration | NLL, Brier, 15-bin ECE | `nll`, `brier`, `ece_15`; state ECE bin convention from code |
| Error ranking | AUROC/AUPRC with error as positive; step-integral AURC | `error_auroc`, `error_auprc`, `aurc` |
| Retrospective tradeoff | Whole-tie-group risk/recall-versus-coverage and coverage at 95% observed accuracy | Native curves or recomputation from saved predictions/confidences |
| OOD admissions | SVHN admissions / SVHN examples | `ood_false_admission_rate`; no CIFAR accuracy |

Retrospective curves and `coverage_at_95pct_accuracy` use evaluation labels for
analysis. They do not define deployable thresholds or replace frozen-rule results.
Fixed-coverage points must show actual attainable coverage when ties prevent the
nominal budget. Abstentions count as false negatives for recall. Unobserved true
classes have undefined recall and are excluded from macro recall; no-prediction
class precision follows the notebook's zero convention.

Wilson intervals describe admitted-example binomial uncertainty conditionally on
the fitted rule; they do not include training-seed variation or turn adaptive
calibration selection into a distribution-shift guarantee. Native collector rows
may lack intervals/counts; obtain them from verified per-image exports before
claiming a complete results table. Do not fill missing evidence with zero.

Any future interval for a *difference* between methods must use paired base-image
IDs. For corruption summaries, resample each base image with all of its condition
outputs together. This paired/bootstrap analysis is proposed here, not implemented
or validated by the existing collector. Until it is implemented and checked,
report descriptive differences without significance claims. With one seed, do
not report mean/spread over seeds or broad robustness to initialization.

## Results tables and figure plan

| ID | Intended display | Required evidence and grouping |
| --- | --- | --- |
| T1 | Run inventory and recipe matrix | Source/driver hashes, checkpoint IDs, splits, class counts, seeds, completion/exclusions |
| T2 | Clean probability and classification comparison | One block per checkpoint/label-space/track/population; top-1, NLL, Brier, ECE, error AUROC/AUPRC, AURC |
| T3 | Calibration-frozen selective results | Method/rule, calibration support, evaluation/admitted counts, coverage, accepted accuracy and Wilson interval, confidently wrong rate; no-region retained |
| T4 | Corruption results | All 75 conditions per block; T3 quantities and target shortfalls; descriptive summaries labelled explicitly |
| T5 | Far-OOD and dataset checks | Separate OOD false admissions/AUROC panel and six-dataset in-distribution panel; no combined accuracy rank |
| F1 | Data-flow/protocol diagram | Training, validation selection, calibration freeze, clean/shift evaluation boundaries |
| F2 | Clean risk/recall versus coverage | Same checkpoint and population; full tied groups; .95 reference; frozen rule marked separately |
| F3 | Reliability and coverage versus corruption severity | Facet by corruption/backbone/track; show reliability and coverage together; preserve no-admission gaps |
| F4 | OOD rejection versus clean coverage | Paired frozen rules with ID population stated; no threshold tuning from this plot |
| F5 | Confusion and failure examples | Per-class recall and directed confusions; predeclared selection such as largest error counts; exemplar IDs and condition recorded |

Use `experiment_results.csv` for compatible aggregate blocks and
`experiment_inventory.csv` for status/exclusions. The native exports and
`matched_shift_metrics.csv` plus per-image `.npz` files supply counts, curves,
class detail and case studies absent from the aggregate table. Cross-check row
counts, method keys and identities before interpreting any display. These are
planned displays; no completed figures or blank tables are research evidence.

## Write-up outline

1. **Abstract.** State the question, both backbone families, selective-prediction
   target and evaluation conditions. Add measured findings only after the results
   gate below; include the strongest limitation and any negative finding.
2. **Problem and motivation.** Explain confident errors, abstention and the
   coverage/review-workload tradeoff. Treat operational settings as motivation,
   not deployment or safety validation of CIFAR experiments.
3. **Related work.** Position softmax confidence, temperature scaling, selective
   prediction and SDM using the existing project references. Verify citations and
   any comparative claim against the original papers before final submission.
4. **Research questions and methods.** Use the questions, method distinctions,
   exact recipes and split table above. Explain pretraining exposure, final-epoch
   versus validation-best checkpoints, calibration budgets and .95 admission rules.
5. **Experimental protocol.** Give source/checkpoint identities, environment,
   transforms, freezing order, condition inventory, metric definitions, uncertainty
   conventions and reproducible commands from the matched-evaluation guide.
6. **Results.** Present T1–T5 and F2–F5 in order: clean baseline, frozen selective
   behavior, corruption degradation, OOD, then calibration-budget/dataset checks.
   Distinguish completed observations, pipeline evidence and missing runs.
7. **Failure analysis and discussion.** Inspect no-region outcomes and class
   confusions; distinguish inadequate confidence ranking, probability calibration,
   label-space difficulty and learned-adaptor prediction changes. These are
   candidate explanations to test, not established causes from one run.
8. **Limitations and responsible interpretation.** Cover one seed, unequal
   architectures/support/pretraining, non-independent corruption images, historical
   test exploration, calibration-policy differences and absence of deployment
   evidence. Fine/coarse results answer different classification questions.
9. **Conclusion.** Answer each research question with its actual scope, including
   null/mixed results. State whether useful coverage and reliability co-occurred,
   and what remains untested.
10. **Reproducibility appendix.** Include manifests, run inventory, exact commands,
    full condition tables, class-level results, definitions, exclusions and the
    distinction between running-campaign and later checkpoint-runtime revisions.

## Results gate and review responsibilities

Populate final results only when each included run is complete, provenance/replay
checks pass, methods and populations match, and missing/no-region cases remain
visible. Reconcile all tabulated values with their underlying masks/counts and
regenerate plots from the verified exports. Audit captions and abstract claims
against the resulting evidence before submission.

Randell's code-review order is PR #27 and PR #30, then PR #29 (depends on #27) and
PR #23 (depends on #27/#30). The PR descriptions are the authoritative review
checklists and validation records. Ready for review is not independent approval,
merge, completed research or closure of #6. The Allen Schmaltz meeting remains
unconfirmed in #6. No author, speaking role, deadline or team decision is assigned
by this outline.

See [matched evaluation and collection commands](matched-shift-evaluation.md).

The [research write-up draft](research-writeup.md) contains the narrative methods
and dated preliminary clean results, with its own aggregate evidence snapshot.
