# Project scope and evidence

The position paper proposes recognition of **one isolated printed Persian letter** as one of 32 alphabet classes. The uploaded Arshasb sample instead contains connected Persian text with **word** annotations. These are complementary development resources, not interchangeable training labels.

## Prototype boundary

| Paper objective | Prototype scope | Remaining full study |
| --- | --- | --- |
| Verified, balanced corpus of 32 letters | A canonical alphabet map and a separately identified synthetic-letter development corpus; inspect the supplied Arshasb word annotations. | Collect and independently review the proposed real-camera corpus; verify every class and freeze grouped partitions. |
| Compact CNN compared with HOG + SVM and pretrained MobileNetV2 | All three were trained and compared on common synthetic partitions, with seed 42. MobileNetV2 was selected by validation; held-out test accuracy was 60.94%. | Repeat across three seeds, collect camera images, and run the synthetic-only versus mixed-camera ablation. |
| Application showing character, name, and score | Local browser workflow for dataset review and a single-letter crop from an uploaded or captured image. | Validate usability and input quality on actual classroom phones and computers. |
| Controlled recognition evaluation | Report synthetic, camera, and dataset-annotation results separately, with explicit sample counts. | Evaluate a frozen selected model once on the untouched camera test partition and measure latency on the demonstration computer. |

This table defines the intended boundary; a model or experiment is complete only when its saved artifact and measured report exist. A synthetic benchmark does not establish performance on the supplied pages or on phone photographs. The full paper's study is broader than this initial prototype.

## What the uploaded dataset establishes

The supplied sample contains **100 pages, 18,994 annotated words, 1,905 lines, and 3,737 distinct raw word strings**. Its annotations associate word transcriptions with page coordinates; they do not identify individual character boxes. Arshasb uses the Far_ketab font, so this sample alone also cannot establish generalization across fonts.

Use its page overlays and word crops to inspect labels, learn the script, and review annotations. Do not split a connected word into guessed character boxes and present those guesses as ground truth. Contextual forms, ligatures, detached dots, and joined strokes make that a separate annotation or sequence-OCR task.

Keep original word strings intact. The dataset audit found eight duplicate-crop label conflicts associated with trailing zero-width non-joiners (U+200C). Raw-label agreement and agreement after an explicitly documented normalization policy are different results. Preserve both views; do not silently remove characters to improve a score. Source annotations are reference labels, not independent evidence of recognition quality.

## Accuracy and confidence

The user aims for 100% recognition on the given dataset. This is a goal to measure, not a result or a guarantee. Displaying an existing annotation, retrieving a previously known image, or achieving 100% training accuracy does not measure OCR generalization. A finite held-out test can score 100% while future images still fail.

Keep automatic predictions, model scores, and human corrections separate. A reviewed final annotation can be correct even when the original prediction was wrong; report automatic accuracy before correction. Model scores are uncalibrated confidence indicators unless a separate calibration study establishes otherwise. They are not measured probabilities that a prediction is correct. If uncertain predictions are rejected, report full-set accuracy plus accepted-image accuracy and coverage.

The paper's actual engineering targets are **at least 90% camera-test accuracy, macro F1 of 0.90, per-class recall of 0.80, and 95th-percentile recognition time below two seconds**. Its primary evaluation requires **384 held-out camera images**, twelve per class. Those photographs were not supplied with Arshasb. Report accuracy, macro F1, per-class precision/recall, confusion counts, sample counts, and an uncertainty interval; separate camera and synthetic results. A small classroom demonstration is additional evidence, not a replacement for that test.

## Classroom capture recommendation

Start with the web application running on a computer. Upload one complete isolated letter, including all dots, and inspect the crop before recognition. A phone can photograph the letter and upload the image through the browser or transfer it to that computer. This allows the same interface and model to serve both devices.

Live browser camera access depends on browser support, permission, and a secure context; remote `getUserMedia` generally needs HTTPS. Phone access over a local network requires user-configured network binding, access controls, and appropriate HTTPS if live capture is needed. Do not expose the development server publicly. Remote hosting is a separate deployment decision; it is not required for this prototype.

## Basis in the attached paper

PDF pages 2 and 4 define the isolated-letter scope and objectives. Pages 5–7 describe corpus collection, group separation, dot-preserving preprocessing, and model comparisons. Page 8 gives evaluation targets and distinguishes demonstration results from statistical evaluation. Page 9 describes the local application and frozen deployment artifacts. The paper describes planned specifications, not completed experiments.

Follow [COLLECTION_PROTOCOL.md](COLLECTION_PROTOCOL.md) before adding real images or claiming camera performance.
