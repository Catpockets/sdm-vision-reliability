# Preventing Catastrophic Hallucinations in LLMs and Vision Models

This document preserves the initial project idea and requirements supplied by the team. Claims in the original abstract are motivation to investigate, not results established by this repository. In particular, reliability under distribution shift must be evaluated; a universal accuracy guarantee is not assumed.

## Author/s

Professors | Allen Schmaltz (Harvard Computer Science PhD & Postdoc, Reexpress AI Founder)

## Abstract

State-of-the-art vision models routinely boast high headline benchmark accuracies (e.g., 90–95%), yet in real-world deployment they suffer from severe overconfidence and silent hallucinations, where mistakes on corrupted or out-of-distribution (OOD) inputs result in catastrophic, undetected failures due to standard softmax overconfidence. To address this fundamental limitation, this project builds on recent breakthrough research by Allen Schmaltz (former Harvard researcher and founder of Reexpress AI), who introduced the Similarity-Distance-Magnitude (SDM) framework to reframe deep networks as hidden instance-based metric learners. In discrete NLP benchmarks (such as automated fact-checking and sentiment classification), SDM decomposes predictive uncertainty into three interpretable geometric signals—Similarity (depth-wise exemplar matches), Distance (normalized $L^2$ training margin), and Magnitude (decision-boundary logits)—to construct a High-Reliability region that guarantees target conditional accuracy (e.g., $\alpha \ge 0.95$) by selectively abstaining from uncertain inputs. The primary focus of this project is to expand SDM beyond binary NLP settings to multi-class vision classification ($\vert Y\vert \ge 100$), investigating whether feature-space geometry can eliminate catastrophic visual errors by maintaining guaranteed $\ge 95\%$ accuracy on admitted instances while reliably rejecting degraded and out-of-domain inputs across CIFAR-100, CIFAR-100-C, and SVHN benchmarks. As an optional extension, students can also adapt SDM as a test-time guardrail over frozen multimodal hidden states (e.g., LLaVA, Qwen-VL) to detect and suppress object hallucinations on the POPE benchmark, providing hands-on experience at the intersection of conformal prediction, metric learning, and actionable interpretability-by-exemplar.

## Links

- [Video overview](https://youtu.be/bKswgsyRAPo)
- [Allen Schmaltz’s website](https://allenschmaltz.github.io/)
- [Local research reference and poster](allen-schmaltz-sdm.md)
- [SDM research repository](https://github.com/ReexpressAI/sdm_activations)
- [Project GitHub](https://github.com/Catpockets/sdm-vision-reliability)

## Datasets & Benchmarks

- **CIFAR-100 / CIFAR-10** — Core multi-class vision benchmarks for in-distribution evaluation, 60k images across 100 categories.
- **CIFAR-100-C** — Robustness benchmark featuring 15 corruption types to test distribution-shift resilience (Hendrycks & Dietterich, 2019).
- **SVHN** — Street View House Numbers; standard far-OOD dataset for testing generalization (Netzer et al., 2011).
- **POPE** — Benchmark for quantifying and mitigating visual object hallucinations in VLMs (EMNLP 2023).

Clarification: the original dataset description above groups CIFAR-100 and CIFAR-10 together; the 100-category scope refers to CIFAR-100. CIFAR-10 is an optional smaller pilot, outside the core 100-class target.

## Papers to Read

These are the reading topics and attributions from the initial proposal; bibliographic verification is part of the starter literature-review task.

- Uncertainty decomposition via SDM geometric signals | Schmaltz, ACL 2026
- Risk-coverage trade-offs and selective prediction mechanics | Geifman & El-Yaniv, NeurIPS 2017
- Temperature scaling and neural network overconfidence | Guo et al., ICML 2017
- Adaptive prediction sets for distribution-free uncertainty | Angelopoulos et al., ICLR 2021
- Scalable Bayesian last-layer uncertainty estimation | Harrison et al., ICLR 2024
- Foundational vision transformer (ViT) architectures | Dosovitskiy et al., ICLR 2021

## Team

Puya[1], Kandy[3], Jeff[3], Yiwen[1], Karim[1]

Bracketed annotations are retained from the supplied proposal; their meanings are not yet documented.

## Presentation Links

- Week 5 Presentation: TBD
- Week 10 Presentation: TBD
- Week 14 Presentation: TBD

## Web Deliverable

TBD
