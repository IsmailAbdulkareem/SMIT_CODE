# Assignment 11a — Multi-class Animal Classification (CNN)

Multi-class image classification of 5 animal species using transfer learning with **ResNet50**.

## Files

- `multiclassificaiton-animals.ipynb` - source code / solution notebook

## About the Dataset

- Source: [animal-image-classification-5-species](https://www.kaggle.com/datasets/miadul/animal-image-classification-5-species) (Kaggle)
- 5 animal classes loaded via `image_dataset_from_directory`
- Split into 80% training / 20% validation

## Pipeline Used

1. Dataset downloaded with `kagglehub`.
2. Loaded images with TensorFlow's `image_dataset_from_directory` (224x224, batch size 32).
3. Performance optimization with `cache()`, `shuffle()` and `prefetch(AUTOTUNE)`.
4. **Data augmentation:** random flip, rotation, zoom and contrast.
5. **Transfer learning:** ResNet50 backbone with ImageNet weights, frozen during training.
6. **Classifier head:** GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.3) → Dense(5, softmax).
7. Compiled with Adam (lr=0.001) and `sparse_categorical_crossentropy`.
8. Callbacks: `EarlyStopping` (patience 4) and `ReduceLROnPlateau` (factor 0.5).
9. Trained on GPU for up to 12 epochs.

## How to Run

1. Install requirements: `pip install tensorflow kagglehub jupyter`
2. Open the notebook in Jupyter and run all cells (`Kernel -> Restart & Run All`).