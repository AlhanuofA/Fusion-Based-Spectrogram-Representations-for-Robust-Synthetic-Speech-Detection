# Fusion-Based-Spectrogram-Representations-for-Robust-Synthetic-Speech-Detection
This repository contains the implementation used in the paper [Fusion-Based Spectrogram Representations for Robust Synthetic Speech Detection].


The study evaluates three image-based input representations for Synthetic
Speech Detection (SSD):

1. **Baseline:** Log-Mel Spectrogram
2. **Variant I:** Dual-Representation Stacked Fusion
   (Log-Mel + Wav2Vec 2.0)
3. **Variant II:** Tri-Channel Spectro-Contextual Fusion
   (Log-Mel + Delta + Wav2Vec 2.0)

## Repository Structure

- 01_Data_Preparation_and_Feature_Representations.ipynb
  - Baseline Log-Mel generation
  - Variant I generation
  - Variant II generation
- 02_Model_Training_and_Evaluation_YOLO.ipynb
- 03_Model_Training_and_Evaluation_Other_Models.ipynb
  - ResNet50
  - ConvNeXt-XLarge
  - EfficientNet-B4


## Dataset Structure

Audio/

├── Train/

│   ├── bonafide/

│   └── spoof/
├── Val/
│   └── ASV19LA/
│       ├── bonafide/
│       └── spoof/
└── Test/
    ├── ASV19LA/
    ├── ASV21DF/
    └── In-the-Wild/

Each evaluation dataset contains separate `bonafide` and `spoof` directories.

## Datasets

The following publicly available datasets were used:

- ASVspoof 2019 Logical Access (LA)
- ASVspoof 2021 Deepfake (DF)
- In-the-Wild Audio Deepfake Dataset


## Input Representations

### Baseline — Log-Mel Spectrogram

The baseline converts each preprocessed audio sample into a Log-Mel spectrogram.

### Variant I — Dual-Representation Stacked Fusion

Variant I combines a Log-Mel spectrogram with contextual representations extracted using the pretrained `facebook/wav2vec2-base` checkpoint.
The Wav2Vec 2.0 representation and Log-Mel spectrogram are vertically stacked to form a single grayscale representation.

### Variant II — Tri-Channel Spectro-Contextual Fusion

Variant II combines three complementary representations:

- **R:** Log-Mel spectrogram
- **G:** Delta coefficients
- **B:** Wav2Vec 2.0 contextual map

The three feature maps form an RGB-format input representation. The channels represent different feature spaces rather than natural color information.


Note: Wav2Vec 2.0 is used as a frozen pretrained feature extractor. No self-supervised training or fine-tuning of Wav2Vec 2.0 is performed in this work.


## Generated Data

The representation-generation notebook produces separate image directories for:

- `Images` — Baseline
- `ImagesV2` — Variant I
- `Images_Fusion` — Variant II

## Training and Evaluation

All models are trained using the ASVspoof 2019 LA training set and evaluated using the datasets described in the paper.
See the corresponding notebooks for the exact training configurations and evaluation procedures.
