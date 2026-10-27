# DeepFER: Facial Emotion Recognition Using Deep Learning

A leakage-free PyTorch pipeline that classifies 48x48 grayscale face images into seven emotions: **angry, disgust, fear, happy, neutral, sad, surprise**. It compares a classical baseline, a custom CNN and a fine-tuned pretrained ResNet18, and combines the two networks into an ensemble.

> **Headline result:** the ensemble of the tuned CNN and tuned ResNet18 reaches **68.5% test accuracy and 0.653 macro-F1** on an untouched test set, against 27.1% for always predicting "happy" and about 34% for a Random Forest on hand-crafted image statistics.

---

## Table of contents
1. [Project overview](#project-overview)
2. [Repository contents](#repository-contents)
3. [Dataset and data audit](#dataset-and-data-audit)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Key insights](#key-insights)
7. [How to run](#how-to-run)
8. [Inference with the saved model](#inference-with-the-saved-model)
9. [Limitations and ethical considerations](#limitations-and-ethical-considerations)
10. [Future work](#future-work)
11. [Author](#author)

---

## Project overview

Facial expressions are a key signal of human emotion, with applications in human-computer interaction, mental-health monitoring and customer service. The task is hard because expressions differ subtly, faces vary in pose, lighting, age and occlusion, and the labels are noisy.

The project covers the full workflow: data audit and cleaning, exploratory analysis, hypothesis testing, preprocessing, three model families with hyperparameter tuning, evaluation, explainability (Grad-CAM) and a saved, reloadable model with a latency benchmark.

**Tech stack:** Python, PyTorch, torchvision, scikit-learn, OpenCV, pandas, NumPy, SciPy, Matplotlib, Seaborn.

## Repository contents

```
.
├── Facial_Emotion_Recognition_Using_Deep_Learning_EDA.ipynb   # data audit, cleaning, 22 charts, business recommendations
├── Facial_Emotion_Recognition_Using_Deep_Learning_ML.ipynb    # preprocessing, hypothesis tests, 3 models, evaluation, export
├── deepfer_best.pt                                            # saved best single network (tuned ResNet18), optional
├── requirements.txt                                           # see the dependencies section below
└── README.md
```

The dataset itself is **not** stored in this repository (see below).

## Dataset and data audit

The dataset is a FER-2013-style collection of **48x48 grayscale face crops**, organised as `train/` and `validation/` folders with one sub-folder per emotion.

An audit of the raw data found several problems that were fixed before modelling:

| Finding | Detail |
|---|---|
| Duplicate archive copy | The zip has 71,774 files, but only **35,887 are unique images**: a nested `images/images/` folder is an exact copy of `train/` + `validation/` and is ignored |
| Redundant copies | 1,853 identical images inside the real data |
| Train/validation leakage | 510 identical images appear in both splits (520 validation images removed) |
| Conflicting labels | 57 groups of identical images carry different labels (160 images removed) |
| Near-blank frames | 2 images removed |
| Corrupt files / missing values | None |
| **Clean dataset** | **33,975 images** (27,501 train, 6,474 validation), 5.3% removed |

Duplication was not random: about **33% of surprise** and **28% of disgust** images were in duplicate groups, against roughly 4-5% for happy, neutral and sad (chi-square test, Cramer's V = 0.31). The classes are also strongly imbalanced: **happy is about 19x larger than disgust**.

**Splits used for modelling:** the provided `validation/` folder is kept as an untouched **test set** (6,474 images). A stratified 15% of the training images becomes the validation set used for tuning, early stopping and model selection (23,375 train / 4,126 validation / 6,474 test).

## Methodology

**Preprocessing and augmentation**
- Pixel standardisation with training-set statistics (ImageNet statistics for the pretrained model; grayscale replicated to 3 channels and resized to 96x96)
- Horizontal flip, rotation up to 15 degrees, shifts up to 10%, scaling 0.9-1.1, brightness and contrast jitter (chosen from the EDA, label-preserving)
- Class imbalance: class-weighted cross-entropy with softened inverse-frequency weights, plus label smoothing (0.1)

**Models**
1. **Random Forest** on hand-crafted image statistics (brightness, contrast, log-sharpness, edge density, asymmetry, upper-lower difference, pixel range), tuned with RandomizedSearchCV and 3-fold cross-validation. Used as a baseline to show that global statistics cannot separate emotions.
2. **Custom CNN** (about 1.2M parameters): four blocks of 2x [Conv-BatchNorm-ReLU] + MaxPool + Dropout2d (32, 64, 128, 256 filters), global average pooling and a small classifier. Small hyperparameter search over learning rate, dropout and weight decay.
3. **ResNet18 transfer learning:** ImageNet-pretrained backbone, new head trained first with the backbone frozen, then the whole network fine-tuned. Fine-tuning learning rate searched over 1e-4, 3e-4 and 1e-3.

**Training:** AdamW, one-cycle learning-rate schedule, mixed precision, gradient clipping and early stopping on validation macro-F1.

**Extras:** flip test-time augmentation, soft-voting ensemble of the two networks, Grad-CAM, error analysis, checkpoint export and latency benchmark.

**Model selection** uses validation macro-F1 only. The test set is reported once and never used to choose a model.

## Results

All numbers are on the untouched test set (6,474 images). Always predicting "happy" gives 27.1% accuracy.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (tuned) | about 34% | about 0.27 |
| Custom CNN | 63.4% | 0.565 |
| Custom CNN (tuned) | 65.1% | 0.599 |
| Custom CNN (tuned) + flip TTA | 65.6% | 0.602 |
| ResNet18 | 67.1% | 0.633 |
| ResNet18 (tuned) | 67.0% | 0.635 |
| ResNet18 (tuned) + flip TTA | 68.1% | 0.649 |
| **Ensemble (CNN + ResNet18, flip TTA)** | **68.5%** | **0.653** |

The ensemble was selected on validation macro-F1 (0.667). Its advantage over the single tuned ResNet18 with flip TTA is small (about 0.4 points of accuracy), which is within the noise of a test set this size. Run-to-run variation with different random seeds should be expected.

**Per-class scores of the tuned ResNet18 (test set):**

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| happy | 0.889 | 0.872 | 0.881 | 1,752 |
| surprise | 0.739 | 0.736 | 0.737 | 568 |
| neutral | 0.611 | 0.687 | 0.646 | 1,182 |
| angry | 0.564 | 0.620 | 0.591 | 893 |
| disgust | 0.661 | 0.500 | 0.569 | 82 |
| sad | 0.563 | 0.553 | 0.558 | 1,083 |
| fear | 0.511 | 0.421 | 0.462 | 914 |

Disgust has only 82 test images, so its scores are noisy.

**Most frequent confusions (true -> predicted):** sad -> neutral (206), fear -> sad (171), fear -> angry (159), neutral -> sad (141), sad -> fear (128), angry -> neutral (114).

**Latency:** [fill in from the notebook, e.g. X ms per image on a T4 GPU]

## Key insights

- Simple global image statistics (brightness, contrast, edge density and others) carry very little information about emotion. Brightness is the strongest cue but explains only 4.4% of the variance, and a Random Forest on them reaches about 34% accuracy. Convolutional networks that learn local patterns are needed.
- Surprise images are brighter than sad images about 71% of the time (Mann-Whitney U test), a possible shortcut that brightness augmentation helps to reduce.
- Data quality mattered: without removing duplicates and train/validation leakage, evaluation scores would have been inflated.
- Transfer learning gave the clearest single gain (ResNet18 test macro-F1 0.635 vs 0.599 for the tuned CNN trained from scratch).
- Grad-CAM shows the model concentrates on the central face and not on the background or watermarks. The maps are coarse (3x3 final feature map), so they show where the model looks but not which fine details drive a decision.
- Many remaining errors involve ambiguous or arguably mislabelled faces, head tilt, closed eyes, glasses or poor image quality, so accuracy has a ceiling well below 100%.

## How to run

The notebooks are designed for **Google Colab**. Use a **GPU runtime** for the ML notebook (Runtime > Change runtime type > T4 GPU). The EDA notebook runs on CPU.

1. Place the dataset zip somewhere in Google Drive and unzip it so that you have a folder containing `train/` and `validation/` (each with the seven emotion sub-folders).
2. Open a notebook in Colab and mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Copy the dataset to Colab's local disk, which is much faster than reading from Drive (change `SRC` to your folder):
   ```python
   import shutil, os
   SRC = '/content/drive/MyDrive/YOUR_FOLDER_NAME/images'
   os.makedirs('/content/images', exist_ok=True)
   for s in ['train', 'validation']:
       shutil.copytree(f'{SRC}/{s}', f'/content/images/{s}')
   %cd /content
   ```
   The notebooks expect a folder named `images` in the working directory. If you have the original zip instead, upload `Face_Emotion_Recognition_Dataset.zip` next to the notebook and the loading cell extracts it for you.
4. Run all cells (*Runtime > Run all*). The ML notebook trains for roughly 20-40 minutes on a T4 GPU.
5. Copy the saved model to Drive before the session ends:
   ```python
   !cp deepfer_best.pt "/content/drive/MyDrive/YOUR_FOLDER_NAME/"
   ```

**Dependencies** (all preinstalled in Colab): `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `opencv-python`, `matplotlib`, `seaborn`, `pillow`.

## Inference with the saved model

`deepfer_best.pt` contains the weights of the tuned ResNet18 together with the class names and the preprocessing settings. It does **not** contain the ensemble. The ML notebook defines `load_checkpoint` and `predict_image`, which you can reuse:

```python
model, ck = load_checkpoint('deepfer_best.pt')
probs = predict_image(model, ck, 'path/to/face.jpg')     # array of 7 probabilities
print(ck['classes'][probs.argmax()], probs.max())
```

The input should be a tightly cropped face. To process full camera frames, a face detector (for example an OpenCV Haar cascade or MediaPipe) has to be added in front of the classifier. A live webcam application is not included in this repository.

## Limitations and ethical considerations

- The model recognises **facial expressions, not a person's true feelings**. Expressions vary across individuals and cultures, and people may hide or fake them.
- Labels are noisy and images are only 48x48 pixels, so there is an accuracy ceiling. Fear versus sad/angry and sad versus neutral are the weakest areas.
- All data comes from a single source. Performance on other cameras, populations, poses and lighting is likely lower and was not tested. Fairness across age, gender and skin tone was not evaluated.
- Disgust has very few examples, so its estimates are uncertain.
- Faces are sensitive personal data. Any real use needs consent, no unnecessary storage of images, and the system should support human decisions rather than make high-stakes ones alone.

## Future work

- A larger, cleaner and better-labelled dataset, especially for disgust and fear
- Higher-resolution inputs and stronger robustness to lighting, pose and occlusion
- Face detection, temporal smoothing over video frames and a live demo (for example Gradio or Streamlit)
- Model compression (ONNX, quantisation) for edge deployment
- Evaluation on an external dataset and a fairness analysis

## Author

**Harsh Raj**  
[GitHub](https://github.com/your-username) | [LinkedIn](https://www.linkedin.com/in/your-profile)

*Edit the author line, links and add a license (for example MIT) before publishing.*
