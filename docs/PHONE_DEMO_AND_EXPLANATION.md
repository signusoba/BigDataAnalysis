# Phone demonstration and project explanation

The phone is the camera and browser. The Windows computer runs FastAPI and the saved model. Training happens separately, before the demonstration. Uploading a picture does not train the model.

## Access from a different Wi-Fi network

A different network needs a reachable connection to the computer. `127.0.0.1` means the device currently opening that address, so opening it on a phone does not reach the laptop. A laptop's ordinary private LAN address also does not normally work from another Wi-Fi network.

For a temporary classroom demonstration, use **Tailscale Serve**, which shares the app privately between enrolled devices. The current prototype has no application login and supports writing annotations; private access is preferable to publishing its existing endpoints openly.

1. First make the app work on the Windows computer. From the project directory, using the Python 3.12 environment from troubleshooting:

   ```powershell
   .\.venv312\Scripts\python.exe -m uvicorn persian_ocr.api:app --host 127.0.0.1 --port 8000
   ```

   If your correctly configured environment is named `.venv`, use that folder instead. Confirm the local app opens before attempting remote access.

2. Install [Tailscale](https://tailscale.com/download) on the computer and phone. Sign both of your devices into your own Tailscale account and connect them. Other team members should use invited accounts rather than exchanging passwords.
3. Open a second PowerShell terminal on the computer and run:

   ```powershell
   tailscale serve --bg 8000
   tailscale serve status
   ```

   If `tailscale` is not recognized, open a new terminal after installing it. Follow Tailscale's setup prompt if HTTPS must be enabled for your private network.

4. Copy the HTTPS address printed by Tailscale and open it in the phone's browser while Tailscale is connected. Use the actual printed address; this guide does not create one.
5. Keep the computer awake, the application process running, and Tailscale connected. The devices may use different Internet connections. Test on the intended school Wi-Fi before presenting, because network restrictions can affect remote connections.
6. Stop the app with Ctrl+C afterward. Tailscale prints the command for disabling the particular Serve mapping; use that command when the demonstration is finished.

Official reference: [Tailscale Serve](https://tailscale.com/kb/1312/serve). Serve's CLI supports a local port target and background operation. These instructions have not been exercised on the team's physical devices or school network. No tunnel or public deployment was created by this project task.

If the professor needs an ordinary link without installing Tailscale, a hosted HTTPS deployment with access control is a separate option. It must include the trained artifacts and persistent storage for reviewed annotations. Changing Uvicorn's host to `0.0.0.0` alone does not make the computer reachable across the Internet.

## Explain the system by following one image

```text
Phone photo → browser crop → server preprocessing → saved classifier
           → Persian character, letter name, scores, and review guidance
```

1. **Problem:** classify one complete isolated printed Persian letter into 32 canonical classes. Connected words, arbitrary full-page OCR, and handwriting are outside the current validated scope. Similar letters such as ب, پ, ت, and ث show why dot count and location matter.
2. **Input preparation:** correct available orientation, composite transparency on white, convert to grayscale, locate all ink including detached dots, crop, and pad proportionally to 64×64. The pipeline rejects blank or very low-contrast inputs. MobileNet subsequently receives a normalized 128×128 three-channel version. It does not identify individual characters inside an arbitrary word.
3. **Prediction:** load the frozen selected checkpoint, calculate 32 output scores, and map the highest-ranked outputs to Persian Unicode labels and English letter names. Scores are uncalibrated; a score of 0.95 does not establish 95% correctness.
4. **Review:** a person can correct and save an image label. The saved example enters a manifest for a later controlled training run. It does not change the deployed model immediately, and the human correction does not count as a correct automatic prediction.

Relevant implementation files are `preprocessing.py`, `models.py`, `training.py`, `inference.py`, and `api.py` under `persian_ocr/`. The browser code lives under `web/`. The implementation uses PyTorch and FastAPI; the position paper describes a proposal, not a claim that every proposed experiment has already been completed.

## Explain the data and evaluation

Arshasb supplies **100 pages with 18,994 word annotations and 1,905 lines**. Those existing labels support the page viewer and annotation workflow. They do not supply isolated-character bounding boxes and were not relabeled as character examples for training.

The classifier used a separate **5,760-image synthetic corpus** spanning 32 letters, nine font files, and five font families. Each font-file/letter combination has 20 size renderings. These are related renders, not independently collected camera photographs.

| Partition | Images | Purpose |
| --- | ---: | --- |
| Training | 3,840 | Fit model parameters using three font families. |
| Validation | 640 | Choose checkpoints and the deployment model using a different family. |
| Test | 1,280 | Measure held-out performance using another font family. |

All weights and sizes of a font family stay in one partition. This reduces source leakage. Selection used validation macro F1, which gives equal importance to every letter; it was saved before test evaluation. Test results are not a tuning score to repeatedly optimize and still describe as untouched.

| Candidate | What it learns | Validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Compact CNN | Visual filters learned from the letter images. | 41.41% | 28.28% |
| HOG + SVM | Class boundaries over measured edge-direction features. | 69.38% | 40.31% |
| Pretrained MobileNetV2 | A new 32-class head over frozen ImageNet visual features. | 93.44% | 60.94% |

MobileNet was selected by validation macro F1. Its **test macro F1 was 0.5300**, and it correctly classified **780 of 1,280** test images, with **500 errors**. The large validation-to-test drop shows limited generalization to the held-out font. Some classes have zero test recall, so having 32 possible outputs does not mean every letter is reliably recognized.

Camera accuracy has not been measured. The cloud response benchmark also excludes network, upload, and capture time; it is not the response time experienced on a school phone. Full evidence is in `docs/EXPERIMENT_RESULTS.md` and `artifacts/metrics.json`.

## A short explanation in your own words

> Our prototype classifies an image of one isolated Persian letter into 32 classes. The phone provides the image, while our computer runs the trained model and sends back the predicted character and scores. Arshasb has word-level annotations, so we kept its annotation viewer separate from the character classifier and generated an initial letter corpus. We compared a compact CNN, HOG with SVM, and pretrained MobileNetV2 using font-family-separated partitions. MobileNet was selected on validation evidence and achieved 60.94% accuracy on the held-out synthetic test set. This is our measured baseline; we still need independently reviewed camera data and a new reserved camera test set before claiming real-photo performance.

Explain each sentence using an example on screen rather than memorizing the paragraph.

## Demonstration order and questions to prepare for

1. Show one isolated example, then its normalized preview. Explain why losing a dot changes the class.
2. Show the top predictions and explain the difference between a score for one image and accuracy over a labeled test set.
3. Show an incorrect or uncertain result and the confusion matrix. Explain which additional development data might help, without claiming the error has been solved.
4. Open Dataset studio and identify the displayed word text as a supplied reference annotation. Create a reviewed character crop only when it is genuinely an isolated letter.
5. Open Evaluation and distinguish 93.44% validation accuracy from 60.94% held-out test accuracy.

Be ready to answer:

- **Why not train directly on the supplied word boxes?** The task needs one letter and its class per training example. A connected word is a different recognition task.
- **Why choose MobileNet when the paper starts with a compact CNN?** The paper calls for comparison; validation evidence selected the model. MobileNet itself is also a convolutional network.
- **Why is 100% not reported?** The measured automatic test score was 60.94%. Existing labels and corrected outputs cannot replace that measurement, and the current evidence does not establish future-photo accuracy.
- **Does a high confidence score guarantee correctness?** No. It is a model score for one input; a confidently wrong prediction remains an error.
- **Does saving a correction make it learn immediately?** No. The image and reviewed label are stored for later training with controlled grouping and evaluation.
- **What would you improve next?** Collect reviewed photos across all 32 classes, diversify fonts and readable capture conditions, study dot-related confusions on development data, and reserve new camera test images before further tuning. The paper's camera experiment and repeated-seed study remain future work.
