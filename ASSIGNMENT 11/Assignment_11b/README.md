# Assignment 11b — Gun vs No Gun Detection (CNN)

Binary image classification (gun / no gun) using a custom CNN built with Keras.

## Files

- `Assignment_11b(Gun_vs_noGuns).ipynb.txt` - source code / solution notebook

## About the Dataset

- Source: [gundetection](https://www.kaggle.com/datasets/atulyakumar98/gundetection) (Kaggle)
- Images labelled `1` (gun) when a non-empty `.txt` label file exists for the image, otherwise `0` (no gun).
- Split into 75% train / 25% test with a stratified split (`random_state=101`).

## Pipeline Used

1. Dataset downloaded with `kagglehub` and all `.jpg` paths collected with `glob`.
2. Labels extracted from the accompanying `.txt` files.
3. Images decoded, resized to 224x224 and normalized to `[0, 1]` via a `tf.data` pipeline (batch size 64, prefetch).
4. **Custom CNN:** 4 Conv2D blocks (32 → 64 → 128 → 256) each with BatchNormalization and MaxPooling, then Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(1, sigmoid).
5. Compiled with RMSprop (lr=0.0005) and `binary_crossentropy`.
6. `EarlyStopping` callback (patience 3) applied during training (up to 15 epochs).

## How to Run

1. Install requirements: `pip install tensorflow kagglehub scikit-learn jupyter`
2. Open the notebook in Jupyter and run all cells (`Kernel -> Restart & Run All`).