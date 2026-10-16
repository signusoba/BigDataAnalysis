# Isolated Persian letter collection protocol

Collect one complete, readable, isolated printed letter per image and label it with exactly one of the project's 32 canonical Persian characters. Preserve detached dots and other identifying strokes. Connected words, contextual-form crops, digits, punctuation, and optional handwriting require separate handling; do not add them as unreviewed alphabet classes.

## Planned corpus, not collected totals

The paper proposes 12 font families, five sizes, and two stroke weights: **3,840 synthetic originals**. Eight font families supply training data and two each supply validation and test data. It also proposes **1,920 camera originals**, 60 per class, divided into 36 training, 12 validation, and 12 test photographs per class. This produces 384 camera validation images and **384 camera test images**.

These are collection targets. Record actual accepted counts after review. Prototype font renderings, resized copies, and augmentations must not be reported as collected camera originals. The optional handwriting proposal is 960 images from ten writers; keep its collection, writer-disjoint splits, and scores separate from printed-letter results.

## Manifest

Store collection metadata in UTF-8 CSV or JSONL with the fields below. The training input is **JSONL**, conventionally `data/camera/manifest.jsonl`; convert collection spreadsheets or CSV into that schema before training. Store referenced image files beneath the manifest's directory, using relative paths such as `images/photo_0001.jpg`. Preserve the original image separately from processed model inputs.

| Field | Meaning |
| --- | --- |
| `path` | Relative path to the original image; it must exist within the manifest directory. |
| `label` | One reviewed canonical Persian letter. Use Persian kaf `ک` and yeh `ی`. |
| `split` | Exactly `train`, `validation`, or `test`, assigned before training. |
| `source` | Use `camera` for photographs, `upload` for other uploaded images, `manual_arshasb` for reviewed page crops, or `synthetic_font` for generated letters; keep handwriting distinct. |
| `group_id` | Identifier for all related originals and derivatives that must remain in one partition. |
| `font_family` | Actual printed/rendered font family; record font version and license in source metadata. |
| `specimen_id` | Physical printed specimen identifier for camera images; every recapture shares it. |
| `session_id` | Capture-session identifier, shared by images taken under that session's conditions. |
| `device_id` | Nonpersonal camera/device identifier for camera images. |
| `sha256` | Checksum of the original image for reproducibility and duplicate checking. |
| `verified` | Boolean `true` only after a reviewer checks the image and its label; real images require it. |
| `reviewer` | Nonempty reviewer name or pseudonymous ID; required for reviewed real examples. |

For real camera rows, record `specimen_id`, `session_id`, and `device_id`. For handwriting, record a pseudonymous `writer_id`. Retain an `original_id` or `parent_id` on derived files so they remain traceable. Record the source, collection date, label reviewer, permission/license, and any applicable participant consent in collection metadata; do not place personal information in public manifests. Preserve any raw imported transcription separately from normalized labels and document each normalization rule.

Example schema only; this is not evidence that the example photograph exists:

```json
{"path":"images/photo_0001.jpg","label":"ب","split":"train","source":"camera","group_id":"collection-group-001","font_family":"training-family-01","specimen_id":"sheet-001-cell-02","session_id":"session-001","device_id":"phone-01","verified":true,"reviewer":"reviewer-01"}
```

## Collection and review

1. Verify Persian glyph coverage and permission to use each selected font. Have a second reviewer inspect all 32 label mappings and representative images, especially dot counts and similar shapes.
2. Assign font families, printed specimens, and capture sessions to partitions before generating or photographing them. A test font must not enter training photographs. Photograph distinct specimens and sessions so each partition has all classes without sharing source material.
3. Capture readable variation in illumination, distance, moderate perspective, and background, preferably using at least two devices. Include shadowed examples. Exclude faces, identifying documents, and unrelated personal material from frames.
4. Review image-label agreement, orientation, focus, the entire letter, and detached marks. Correct or reject ambiguous records before setting `verified` to `true`. Log exclusions and class counts.
5. Check exact duplicates and inspect near duplicates before freezing partitions. Record manifest checksums and dataset versions.

## Prevent leakage

All sizes, weights, captures, crops, and transformed versions derived from the same source stay in one partition. Entire font families stay in one partition. Entire capture sessions and physical specimens also stay in one partition. For handwriting, entire writers stay in one partition.

Construct `group_id` so it includes the union of these relationships: if two records share a specimen or session, or connect through another related record, they belong to the same group. A single group must never cross partitions. Do not assume that checking image hashes alone detects related images. Check font, session, specimen, writer where relevant, and parent identifiers as well as hashes before accepting a new dataset version.

Apply augmentations only while loading training images. Use modest readable transformations, never horizontal/vertical flips or alterations that remove or change dots. Preserve aspect ratio and all detached components when preparing crops; selecting only the largest connected component can discard essential marks. Derived images do not increase the count of independent originals.

## Freeze and evaluate

Use training data to fit models and validation data for hyperparameters, preprocessing, model selection, and any uncertainty threshold. Keep the final test data unopened for these decisions. Save a frozen model, label map, preprocessing configuration, dataset version, and threshold before the final test.

Report synthetic and camera results separately. Compare compact CNN, HOG + SVM, and pretrained MobileNetV2 on identical source partitions; include repeated-seed variability and the synthetic-only versus mixed-camera comparison for the full study. Report all errors and rejection coverage. Keep human-reviewed final annotations distinct from raw automatic predictions. Never train on corrections from the reserved test set and then reuse that set as an untouched accuracy measurement.

The paper's camera targets are accuracy ≥0.90, macro F1 ≥0.90, and every class recall ≥0.80. Measure warm end-to-end preprocessing, inference, and label mapping on the demonstration computer, reporting median and 95th-percentile latency with a target p95 below two seconds. Capture/upload time is separate. No collection protocol can guarantee 100% accuracy on future photographs.
