# Pix2Pix GAN for Satellite-to-Map Translation

This project implements a Pix2Pix GAN using PyTorch to generate maps from satellite images. The code is adapted from [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This README outlines the steps to preprocess the dataset, train the Pix2Pix model, and test the model's performance on new images.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Dataset Preparation

The Pix2Pix model is trained on a satellite-to-map dataset, specifically the [Maps dataset](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz). Run the following commands to download and extract the dataset:

```bash
!wget https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
!tar -xzf maps.tar.gz
```

After downloading, we combine and split the images into training and validation sets using the following script:

```python
def combine_and_split_folders(folder1, folder2, output_folder, train_ratio=0.8):
    # Combines images from two folders and splits them into train and validation sets
```

## Model Architecture

The Pix2Pix model consists of a U-Net Generator and a PatchGAN Discriminator:

1. **Generator (U-Net)**: Creates a map from a satellite image.
2. **Discriminator (PatchGAN)**: Distinguishes between real and generated maps by comparing image patches.

The model class definitions are located in the code, with adjustable hyperparameters.

## Training

The model is trained with the following parameters:
- **Input Shape**: (3, 256, 256) RGB images
- **Loss Functions**: Binary Cross-Entropy for the discriminator, and a combination of Binary Cross-Entropy and L1 loss for the generator.
- **Optimizers**: Adam optimizers for both generator and discriminator.

To start training, use the `train_pix2pix` function:

```python
train_pix2pix(dataloader, epochs=30)
```

This function includes a progress bar and displays discriminator and generator loss values for each epoch.

## Testing

To evaluate the model on unseen images, use the `test_pix2pix` function, which visualizes the satellite image, ground truth map, and generated map for comparison:

```python
test_pix2pix(gen, val_dataset, num_samples=5)
```

## Results

Example output images are displayed side by side:
1. Satellite Image
2. Ground Truth Map
3. Generated Map

## How to Run

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Ensure you have the necessary dependencies:
   ```bash
   pip install torch torchvision matplotlib tqdm pillow
   ```
3. **Download the Dataset**: Follow the steps in [Dataset Preparation](#dataset-preparation).
4. **Train the Model**: Run `train_pix2pix` with your dataset to start training.
5. **Test the Model**: Run `test_pix2pix` to see sample results on validation images.

## References

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004)
- [Original Pix2Pix Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
