# Chest-Xray-CNN: image-classification pipeline demo

This is a classroom prototype of a CNN pipeline with a Streamlit app. It is **NOT a medical device** and **NOT a validated pneumonia detector**.

## What it is

The project uses a custom TensorFlow Sequential CNN with three `Conv2D` blocks, global-average pooling, a dense layer, dropout, and a three-class softmax output:

- `INVALID`
- `NORMAL`
- `PNEUMONIA`

The model was trained from scratch and does not use pretrained weights.

Project files:

- `train.ipynb` — data preparation, training, validation, and saved experiment output
- `app.py` — Streamlit inference app
- `requirements.txt` — Python dependencies

## Data

The dataset name and source were **not recorded**, and the images are not included in this repository.

The experiment used 12 original images: 10 `INVALID`, 1 `NORMAL`, and 1 `PNEUMONIA`. Offline augmentation expanded these into 612 images:

| Class | Images |
| --- | ---: |
| INVALID | 210 |
| NORMAL | 201 |
| PNEUMONIA | 201 |

The augmented dataset was then split 80/20 with seed `123`, producing 490 training images and 122 validation images.

## Results

**From saved notebook output, not re-run:** the best epoch, selected by `EarlyStopping` monitoring `val_loss` with patience 5 and `restore_best_weights=True`, gives validation accuracy **1.0000** and validation loss **0.00044693** on 122 validation images.

### How to read these results

Augmentation happened before the split, so augmented copies of the same original image appear in both the training and validation sets. The same validation set drove `EarlyStopping` and `ReduceLROnPlateau`. There is no held-out test set, and precision, recall, F1, and a confusion matrix were not computed. These numbers do not show how the model performs on new X-rays.

## Model hosting

`model.keras` is hosted at [Hugging Face](https://huggingface.co/faiz4320/cnn-model) and downloaded by `app.py` with `hf_hub_download`.

## Known limitations

- Educational prototype; not for medical use
- Tiny independent sample: 1 original `NORMAL` image and 1 original `PNEUMONIA` image
- Augmentation before splitting causes training/validation leakage
- No held-out test set
- Dataset source not recorded
- Class balance is artificial because it was created through augmentation

Planned improvements are to split before augmentation, use a real public dataset, and add precision, recall, F1, and a confusion matrix evaluated on a held-out test set.

## How this was built

Designed and specified by me; implemented with AI coding assistants (Codex and Claude); I review the code, run the experiments and direct the roadmap.

Published on: 2026-10-17

## Credits

- TensorFlow/Keras
- Streamlit
- Hugging Face
