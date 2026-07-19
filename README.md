# CIFAR-10 Image Classifier

A convolutional neural network that classifies CIFAR-10 images into one of 10
categories, built to practice CNNs in TensorFlow/Keras.

## What it does

- Loads the CIFAR-10 dataset (60,000 32x32 color images, 10 classes) via
  `tf.keras.datasets`.
- Normalizes pixel values and builds a small CNN: three `Conv2D` + `MaxPooling2D`
  blocks followed by dense layers.
- Trains with the Adam optimizer and sparse categorical cross-entropy, saving the
  trained model to `cifar10_cnn_model.h5` (and loading it from disk on subsequent
  runs instead of retraining).
- Includes a helper to classify a single image and plot it with its predicted vs.
  true label.

## Tech stack

- Python
- TensorFlow / Keras
- Matplotlib

## Running it

Open `Image Classifier.ipynb` in Jupyter and run the cells in order. Training from
scratch downloads CIFAR-10 automatically and takes a few minutes on CPU.
