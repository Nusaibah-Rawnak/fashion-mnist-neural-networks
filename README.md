# Neural Network Architectures for Fashion-MNIST

A comprehensive evaluation of neural network architectures for image classification on the Fashion-MNIST dataset, comparing MLPs implemented from scratch in NumPy with PyTorch CNNs and transfer learning approaches.

## Project Overview

This project provides a systematic comparison of different neural network architectures for classifying grayscale clothing images:

- **MLPs (NumPy)**: Implemented from scratch to explore forward/backward propagation, initialization, and optimization
- **CNNs (PyTorch)**: Leveraging convolutional layers for spatial feature extraction
- **Transfer Learning**: Fine-tuning pretrained ResNet18 on Fashion-MNIST

## Key Findings

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| MLP (2 hidden, L2 reg) | 85.20% | Best fully-connected model |
| CNN (baseline) | 91.57% | 6.37% improvement over MLP |
| CNN + Dropout (0.4) | **92.40%** | Best overall performance |
| ResNet18 Transfer | 83.26% | Underperformed due to domain mismatch |

### Main Insights

1. **Architectural Impact**: CNNs significantly outperform MLPs (6.37% accuracy gain) by preserving spatial structure
2. **Depth vs. Non-linearity**: Adding one hidden layer to MLPs provides major gains (73.80% → 84.69%), but additional depth shows diminishing returns
3. **Regularization**: Dropout (0.4) effectively reduces overfitting in CNNs, improving accuracy by 0.83%
4. **Data Augmentation**: Harmful for MLPs (-6.77% accuracy) but beneficial for CNNs (+0.65% accuracy)
5. **Transfer Learning**: ResNet18 underperformed custom CNN due to significant domain mismatch between ImageNet and Fashion-MNIST

## Experiments Conducted

### MLP Analysis
- Network depth (0, 1, 2 hidden layers)
- Activation functions (ReLU, tanh, Leaky-ReLU)
- Regularization (L1, L2)
- Normalization effects
- Data augmentation impact
- Hyperparameter search (learning rate, batch size)

### CNN Analysis
- Architecture variations (filter counts, kernel sizes)
- Dropout regularization (0.0 to 0.5)
- Data augmentation
- Hyperparameter tuning

### Transfer Learning
- ResNet18 with 0, 1, 2 fully connected layers
- Domain adaptation challenges

## Dataset

**Fashion-MNIST** contains 70,000 grayscale 28×28 images across 10 clothing categories:
- 60,000 training samples
- 10,000 test samples
- Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Implementation Details

### MLP (NumPy)
- Fully implemented forward/backward propagation
- Custom gradient descent optimizer
- Multiple activation functions
- L1/L2 regularization support

### CNN (PyTorch)
- 2 convolutional layers + 1 fully connected layer
- ReLU activation, max pooling
- Optional dropout regularization
- Adam optimizer

### Hyperparameters
- **MLP**: lr=0.001, batch_size=64, epochs=20
- **CNN**: lr=0.001, batch_size=64, epochs=20
- **ResNet18**: lr=0.001, epochs=5 (limited due to computational cost)

## Results Summary

The best configuration achieved **92.40% test accuracy** using a CNN with 0.4 dropout. Key takeaways:

✅ CNNs are substantially better than MLPs for image classification  
✅ Dropout effectively controls overfitting  
✅ Small kernels (3×3) outperform larger kernels (5×5)  
✅ Transfer learning from ImageNet doesn't always help on specialized domains  

## Project Structure
```
.
├── code.zip            # Complete implementation
├── writeup.pdf         # Detailed report with all experiments and visualizations
└── README.md           # This file
```

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Nusaibah-Rawnak/fashion-mnist-neural-networks.git
cd fashion-mnist-neural-networks

# Extract code
unzip code.zip

# Install dependencies
pip install numpy torch torchvision matplotlib scikit-learn

# Run experiments (instructions inside code files)
```

## Full Report

For detailed methodology, comprehensive results, learning curves, and in-depth analysis, see [writeup.pdf](writeup.pdf).

## Authors

Kazi Ashhab Rahman 
Nusaibah Binte Rawnak

## Acknowledgments

Dataset: Fashion-MNIST by Zalando Research  
Framework: PyTorch, NumPy, Matplotlib

## License

This project is available for educational purposes.
