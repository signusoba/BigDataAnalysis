# Noor · Persian Letter Lab

A working final-project prototype for **recognizing one isolated Persian letter** and **reviewing the supplied Arshasb dataset**. The browser interface works on a computer and accepts phone photographs. It displays a predicted Persian character, its name, ranked model scores, and review guidance.

The attached position paper defines a 32-class isolated-letter project. Arshasb instead supplies full printed pages with word labels. This prototype keeps those tasks separate: the classifier learns from a separately generated isolated-letter corpus; the dataset studio shows existing word annotations and lets a reviewer create genuine character examples. It does not claim full-page or connected-word OCR.

**100% accuracy is a target to evaluate, not a result to assume.** Existing dataset labels, training accuracy, and manually corrected annotations are not automatic recognition accuracy. Phone-camera performance requires an independently collected camera test set. See [project scope](docs/PROJECT_SCOPE.md) and the [collection protocol](docs/COLLECTION_PROTOCOL.md).

## Run the prepared cloud copy

```bash
cd /workspace/BigDataAnalysis
source /workspace/.cloud-setup/bigdataanalysis/env.sh
bash scripts/run.sh
```

The development server uses loopback port 8000 by default. `NOOR_PORT` changes the port. Open the server with your local browser or the host's supported forwarding mechanism. Stop it with Ctrl+C. This project does not publish or expose the server publicly.

## Install a fresh copy

For step-by-step Windows setup, selecting the VS Code interpreter, and the included F5 launch configuration, see [Run in VS Code](docs/RUN_IN_VSCODE.md). The complete ZIP includes the dataset and trained model needed for immediate recognition.

Use Python **3.12** and the CPU package versions provided here. On Linux with Bash:

```bash
bash scripts/setup.sh
source .venv/bin/activate
bash scripts/run.sh
```

The pinned `+cpu` PyTorch wheels target supported Linux/Windows machines. On macOS, install platform-compatible PyTorch/Torchvision versions instead; the included cloud environment is Linux x86-64. On Windows, create the environment with `py -3.12 -m venv .venv`, install with `.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock`, then run `.\.venv\Scripts\python.exe -m uvicorn persian_ocr.api:app --host 127.0.0.1 --port 8000` from the repository root. Direct interpreter commands work without activating a PowerShell script. The lock pins transitive dependencies and verifies artifact hashes; update it deliberately when changing dependencies.

The source repository excludes uploaded data, generated images, and model binaries. The prepared instance and submission ZIP contain these artifacts. A fresh source-only checkout can still show the interface and alphabet; import data and train models using the steps below to enable the corresponding features.

## Use the prototype

1. **Recognize:** choose an image or take a phone photo, drag a crop around one complete printed letter, retain every dot, and request a prediction. Scores are model outputs, not guaranteed probabilities of correctness. A confidence threshold requests review; it cannot reliably detect every out-of-scope word, symbol, or photo.
2. **Review:** select the true letter yourself and provide a reviewer, source group, and dataset split before saving. Predictions never automatically become verified labels.
3. **Dataset studio:** browse the 100 supplied pages, inspect provided word boxes, or manually frame a genuinely isolated letter. The provided word transcription is explicitly labeled as ground truth rather than a prediction. Do not crop an attached contextual letter and label it as an isolated canonical form.
4. **Evaluation:** inspect the actual candidate comparison, held-out results, class confusion matrix, and limitations. Synthetic and real-camera evidence remain separate.
5. **Alphabet:** consult all 32 canonical Persian letters and names.

For a first recognition check, upload `data/letters/images/validation/DejaVuSans_00_00.png` from the prepared copy or ZIP (the synthetic letter `ا`, Alef). This is a development example, not new camera evidence. Do not resave development examples into a different training/test split.

For access from a phone on another Wi-Fi network and a presentation guide explaining the architecture, data, and measured results, see [Phone demo and project explanation](docs/PHONE_DEMO_AND_EXPLANATION.md).

A phone does not train the network. Training runs offline on the computer/server; the browser uploads a crop for inference. Phone access to a computer's local server requires an explicitly configured accessible address or hosting. Image-file camera capture depends on the device; live browser camera APIs generally require HTTPS. Start with desktop uploads and transferred phone photos for the classroom demonstration.

## Import the uploaded dataset

```bash
python scripts/import_arshasb.py --archive /path/to/Arshasb-main.zip
```

The importer also accepts `Arshasb_samples.tar.gz`, rejects archive traversal and links, and preserves existing files. It produces `data/raw/arshasb`. The supplied sample has **100 pages, 18,994 words, and 1,905 lines**, with word and line coordinates. No character boxes are included. `ARSHASB_DATA_DIR` can point the app at another extracted sample directory. Dataset licenses and checksums are retained in the import directory.

## Generate, train, and evaluate

Synthetic generation needs the Noto Arabic and DejaVu font families. On Debian/Ubuntu these are provided by `fonts-noto-core` and `fonts-dejavu-core`. Other machines can supply a font root with `noto/` and `dejavu/` subdirectories using `--font-root`. Inference from an existing trained artifact does not require the font files.

```bash
python -m persian_ocr.corpus --output data/letters
python -m persian_ocr.training --data data/letters --output artifacts --epochs 12 --models cnn hog mobilenet
```

The generated development corpus contains 5,760 synthetic glyph images, not 5,760 camera photographs. All weights and sizes of each font family stay in one split: 3,840 training, 640 validation, and 1,280 test images. Font coverage, file checksums, counts, and provenance are recorded in `data/letters/dataset_manifest.json`. These repeated font renders are correlated examples.

The comparators are the paper's compact three-block CNN, HOG + an RBF SVM, and an ImageNet-pretrained MobileNetV2 with a frozen feature extractor and a trained 32-class head. The small prototype uses seed 42, up to 12 epochs, a batch size of 64, 64×64 shared glyph preprocessing, and 128×128 input for MobileNet. These choices are a starting experiment, not a full replication of the paper's proposed study. The CNN retains the checkpoint with the best validation macro F1. Pretrained weights come from the official PyTorch host with checksum verification.

Selection uses validation macro F1; candidates less than 0.01 apart are compared by validation inference latency. The selection and model checksums are saved **before** test evaluation. Each candidate is then evaluated once on the test partition. Reports include accuracy, macro F1, per-class precision/recall, confusion counts, descriptive Wilson intervals, source-specific outcomes, and latency. `deployment_latency_ms` measures the selected local preprocessing/inference/response path; individual comparison inference timings exclude preprocessing and are labeled accordingly.

Training refuses to replace an existing evaluation unless `--overwrite` is explicit. Use a new output directory for new experiments and do not repeatedly tune to the same test set. The web app reloads matching model metadata after a new trained artifact is saved. `PERSIAN_ARTIFACTS_DIR` selects a different artifact directory.

To incorporate manually reviewed examples from the application:

```bash
python -m persian_ocr.training --data data/letters --additional-manifest data/annotations/manifest.jsonl --output artifacts/mixed-experiment --epochs 12 --models cnn hog mobilenet
```

This creates a different experiment. Camera accuracy is measured only for camera examples in its untouched test partition; saving a photo in the app does not establish that performance. The current UI records one source group per example. Before a full camera study, follow the collection protocol to review font/specimen/session relationships and freeze groups; automated checks only enforce metadata actually supplied.

## Code structure

```text
persian_ocr/
  letters.py          # Canonical 32-class map and label normalization
  preprocessing.py    # EXIF, complete-ink crop, aspect padding; detached dots retained
  arshasb.py          # Safe spreadsheet parsing and source-page adapter
  corpus.py           # Font-disjoint corpus generation and manifest checks
  models.py           # Compact CNN, HOG features, and pretrained comparator
  training.py         # Training, validation selection, locked test reports
  inference.py        # Artifact verification and reviewable predictions
  api.py              # Uploads, predictions, dataset review, annotation/export APIs
web/                  # Responsive interface with pointer-based image cropping
scripts/              # Setup, dataset import, and development server launch
tests/                # Image, split, model save/load, API, and archive checks
docs/                 # Objectives, collection protocol, experiment evidence, attribution
data/                 # Local raw pages, generated corpus, reviewed crops (Git ignored)
artifacts/            # Local model files, checksums, metrics, metadata (Git ignored)
```

Useful APIs are `/api/health`, `/api/overview`, `/api/predict`, `/api/dataset/pages`, `/api/annotations`, `/api/annotations/export`, and `/api/evaluation`. FastAPI also provides its generated API schema.

## Verify and continue the research

```bash
python -m pytest
```

The provided tests exercise invalid uploads, preserving detached dots, split leakage, manual labels, ground-truth provenance, artifact save/load/checksum verification, and safe dataset import. The completed prototype's measured experiment is documented in [EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md).

The remaining research is real-camera collection and independent label review, camera-domain evaluation, additional seeds and the synthetic-vs-mixed ablation, and controlled usability/latency checks on your demonstration devices. Optional handwriting requires its own writer-disjoint dataset. The paper's actual targets are 90% camera accuracy, macro F1 ≥ 0.90, every class recall ≥ 0.80, and p95 local recognition time under two seconds; none can be inferred from synthetic accuracy alone.

See [third-party provenance](docs/THIRD_PARTY.md) before redistributing dataset, font, or model assets.
