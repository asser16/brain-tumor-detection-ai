# Brain Tumor MRI Classification with Transfer Learning

A deep-learning pipeline that classifies brain MRI scans into **four classes** using **transfer learning with ResNet50** (TensorFlow / Keras), built in Google Colab.

> University term project (Artificial Intelligence), December 2025.

## Approach

1. **Data loading**: the MRI dataset is unzipped and organised by class folder (one folder per class).
2. **Exploration**: count images per class and show sample images from each class.
3. **Splitting**: stratified **70 / 15 / 15** train / validation / test split with `scikit-learn`.
4. **Pre-processing and augmentation**: images resized to **224 × 224** and rescaled to [0, 1]. The training set is augmented with rotation (±15°), width and height shifts, zoom and horizontal flips. Validation and test data are not augmented.
5. **Model**: **ResNet50** pre-trained on ImageNet with the convolutional base frozen, plus a custom head:
   `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)`
6. **Training**: Adam optimiser (learning rate 1e-4), categorical cross-entropy, up to 25 epochs with **EarlyStopping** (patience 5, restore best weights) and **ReduceLROnPlateau** (factor 0.3).
7. **Evaluation**: training and validation accuracy and loss curves.

## Tech stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Pillow · Google Colab (T4 GPU)

## Project structure

```
brain-tumor-detection-ai/
├── notebooks/
│   └── brain_tumor_classification_resnet50.ipynb
├── requirements.txt
└── README.md
```

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (or locally with `pip install -r requirements.txt`).
2. Download a 4-class brain MRI dataset (for example the public *Brain Tumor MRI Dataset* on Kaggle, which has glioma, meningioma, pituitary and no-tumor classes) as `archive.zip`.
3. Upload `archive.zip` to `/content/` (or update `zip_path` in the first cell).
4. Run all cells. Use a GPU runtime for reasonable training time.

## Next steps

- Fine-tune the top ResNet50 blocks after the head has converged.
- Report test-set accuracy, a confusion matrix and per-class precision and recall.
- Compare against a custom CNN baseline and lighter models such as EfficientNet or MobileNetV2.

## Author

**Asser Abdelsamad**, Computer Engineering student, Arab Academy for Science, Technology & Maritime Transport (AAST), Alexandria.
