# MNIST Convolutional Neural Network
 This repository contains Python projects for implementing and training CNNs on the MNIST dataset using PyTorch. The projects showcase the application of convolutional layers, pooling, and fully connected layers to classify handwritten digits. Additionally, they explore training with different optimizers and visualize predictions.

## Project Descriptions

### **1. MNIST with Adam Optimizer**
- **Overview**: A CNN is trained to classify the MNIST dataset using the Adam optimizer.
- **Features**:
  - Processes the MNIST dataset with transformations for normalization.
  - Implements a simple CNN with:
    - Two convolutional layers followed by ReLU activation and max pooling.
    - Two fully connected layers, with the final layer predicting one of the 10 digit classes.
  - Trains the model using the Adam optimizer with a learning rate of 0.001.
  - Visualizes predictions after training with random samples from the test dataset.
- **Usage**:
  - Run `MNISTwPIC.py` to train the model, evaluate it, and visualize predictions.

### **2. MNIST with SGD Optimizer**
- **Overview**: A CNN is trained to classify the MNIST dataset using stochastic gradient descent (SGD).
- **Features**:
  - Prepares the MNIST dataset with normalization transformations.
  - Implements the same CNN architecture as the Adam version.
  - Trains the model using SGD with a learning rate of 0.01.
  - Saves the trained model for later use.
- **Usage**:
  - Run `MNIST.py` to train the model and evaluate its performance.

### **Shared CNN Architecture**
- **Model Description**:
  - Two convolutional layers (`Conv2d`) to extract features.
  - ReLU activation for non-linearity.
  - Max pooling to downsample feature maps.
  - A fully connected layer with 128 neurons.
  - An output layer with 10 neurons for classification.
- **Training Features**:
  - Cross-entropy loss function for classification.
  - Options for using Adam or SGD optimizers.
  - Saves trained model weights for future inference or evaluation.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: `torch`, `torchvision`, `matplotlib`

### Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```

### Running the Projects
1. To train with Adam and visualize predictions:
   ```bash
   python MNISTwPIC.py
   ```
2. To train with SGD:
   ```bash
   python MNIST.py
   ```

## Acknowledgments
- Inspired by the MNIST dataset as a benchmark for handwritten digit classification.
- Demonstrates different optimization strategies and their effects on training performance.