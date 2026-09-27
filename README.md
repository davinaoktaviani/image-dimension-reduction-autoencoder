# Image Dimension Reduction with Convolutional Autoencoder

A deep learning project that uses a **Convolutional Autoencoder** to compress grayscale images of **cars and planes** into a lower-dimensional latent representation.

## Problem

The project reduces image dimensionality while attempting to preserve important visual information.

- Dataset: Car and Plane
- Total images: 2,000
- Image size: 28 × 28 grayscale
- Latent dimension: 128
- Compression ratio: approximately 6:1

## Workflow

1. Load car and plane images
2. Normalize pixel values to `[0, 1]`
3. Visualize sample images
4. Split the data into 80% training, 10% validation, and 10% test
5. Train a baseline convolutional autoencoder
6. Train an improved architecture
7. Compare reconstruction quality using SSIM
8. Visualize the learned latent space using t-SNE

## Baseline Autoencoder

```text
Input 1×28×28
    ↓
Conv + ReLU
    ↓
MaxPool
    ↓
Flatten
    ↓
Fully Connected
    ↓
Latent 128
    ↓
Fully Connected
    ↓
Upsampling
    ↓
Conv + ReLU
    ↓
Conv + Sigmoid
    ↓
Reconstructed Image
```

## Improved Autoencoder

The improved model adds:

- A deeper convolutional encoder
- More filters (32 → 64)
- Batch Normalization
- Dropout
- LeakyReLU
- ReduceLROnPlateau
- L2 regularization through weight decay

The improved model has approximately **3.39 million parameters**, compared with **1.62 million** for the baseline.

## Results

| Model | SSIM |
|---|---:|
| Baseline Autoencoder | 0.6930 |
| Improved Autoencoder | **0.6965** |

The improved model increased SSIM by approximately **0.0036**. However, the improvement was small relative to the additional model complexity, and the training curves showed indications of overfitting.

## Latent Space

The improved model produces a 128-dimensional latent representation. t-SNE is used to visualize this representation in 2D.

The visualization shows that the learned representation can separate the Car and Plane classes to some extent, even though the autoencoder is trained for reconstruction rather than classification.
