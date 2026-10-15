# Prototype experiment results

Measured on 15 October 2026. These are actual outputs from the included training pipeline, not targets from the position paper.

## What was evaluated

- **Task:** classify one isolated canonical Persian letter into 32 classes.
- **Data:** 5,760 synthetic images generated from nine font files in five families; no real-camera or handwriting images.
- **Related examples:** the corpus has 288 font-file/letter combinations, each rendered at 20 sizes. These are correlated renders, not 5,760 independent physical specimens.
- **Training:** 3,840 images from Noto Naskh Arabic, Noto Sans Arabic, and Noto Nastaliq Urdu.
- **Validation:** 640 images from DejaVu Sans.
- **Test:** 1,280 images from Noto Kufi Arabic, with 40 images per class. No font family crosses partitions; all 5,760 source image hashes are unique.
- **Experiment:** seed 42; CNN and MobileNet head trained for 12 epochs, batch size 64. MobileNet uses frozen ImageNet-pretrained features. Model selection used validation macro F1 and the documented latency tie rule, then the selection was saved before test evaluation.

## Measured comparison

| Candidate | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Correct test images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact CNN | 41.41% | 0.3365 | 28.28% | 0.2526 | 362/1280 |
| HOG + SVM | 69.38% | 0.6239 | 40.31% | 0.3426 | 516/1280 |
| Frozen pretrained MobileNetV2 | 93.44% | 0.9213 | 60.94% | 0.5300 | 780/1280 |

**Selected model: MobileNetV2.** It correctly classified 780 of 1280 held-out synthetic test images; 500 were incorrect. The descriptive Wilson 95% interval is 58.24%–63.57%. Images share font families and rendering conditions, so this binomial interval does not establish uncertainty across new fonts, devices, or real photographs.

The selected model's validation accuracy was 93.44%, while held-out-font test accuracy was 60.94%. This gap is evidence of limited cross-font generalization. It does not satisfy a 100% target. The small prototype CNN also underperformed; its proposed architecture alone is not evidence of sufficient accuracy.

## Response time

On this cloud instance's CPU, the selected model's warm response path had **9.65 ms median and 11.41 ms p95** over 64 validation inputs. This includes preprocessing, inference, label mapping, ranked scores, and PNG preview generation. It excludes disk loading, network/upload time, and capture. It is not a measurement on the team's phone or classroom computer.

## What this establishes

The code can import the provided pages, generate a reproducible letter corpus, train all three comparators, choose using validation evidence, freeze artifacts, serve predictions, and support reviewed annotation. The model is an initial research baseline, not a fully accurate finished recognizer.

The supplied Arshasb sample has 100 full pages with 18,994 word annotations and 1,905 lines. Those word labels appear in Dataset studio as existing reference annotations. They were **not** used as fake isolated-character labels, and displaying them is **not** an OCR accuracy result. Raw word labels also contain a small number of boundary-ZWNJ conflicts documented in PROJECT_SCOPE.md.

## What remains

1. Collect and independently verify isolated printed-camera letters across all 32 classes. The paper proposes 1,920 photographs, including 384 reserved camera test images. None were supplied or collected here.
2. Increase training font and image-condition coverage, investigate dot-sensitive confusion patterns on development data, and compare additional preprocessing or model changes using validation data.
3. Run the proposed three seeds, camera/synthetic ablation, and device-specific response-time checks. Capture any additional domain evidence separately.
4. Freeze a final configuration before opening a new, reserved camera test set. The synthetic test results above have now been inspected; repeated optimization against them must not be described as untouched final testing.
5. Keep reviewed final annotations separate from automatic predictions. Human correction can produce verified outputs, but it does not turn the measured automatic score into 100%.

## Evidence files

- `artifacts/metrics.json`: full comparison, per-class precision/recall/F1, confusion matrices, descriptive intervals, source counts, training history, and timing.
- `artifacts/selection.json`: validation-only selection and hashes, saved before test evaluation.
- `artifacts/model_metadata.json`: deployed input convention, label map, review thresholds, selected checkpoint hash, and limitations.
- `data/letters/dataset_manifest.json`: generation provenance, source font hashes, counts, and partitions.

The prototype passed **30 automated tests**. Browser checks also exercised image upload, crop review, page word inspection, manual character annotation, UTF-8 CSV export, all 32 alphabet cards, and responsive desktop/mobile layouts. Synthetic-only validation does not substitute for an independent camera test.
