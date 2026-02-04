# CIFAR-10 CNN Classifier

A complete end-to-end Convolutional Neural Network implementation for classifying images from the CIFAR-10 dataset using PyTorch.

## 🎯 Project Overview

This project implements a CNN that achieves **74.64% accuracy** on the CIFAR-10 test set after 10 epochs of training. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

## 📊 Results

- **Test Accuracy**: 74.64%
- **Training Loss**: Decreased from 2.1 → 0.82 over 10 epochs
- **Best Performing Classes**: 
  - Car: 90.9%
  - Truck: 88.8%
  - Plane: 79.2%

## 🏗️ Architecture

The CNN architecture consists of:
- **3 Convolutional Blocks**: Each with Conv2D → BatchNorm → ReLU → MaxPool
- **3 Fully Connected Layers**: With dropout for regularization
- **Total Parameters**: Lightweight design optimized for CPU training

```
Input (32x32x3) 
→ Conv Block 1 (32 filters) → MaxPool
→ Conv Block 2 (64 filters) → MaxPool  
→ Conv Block 3 (128 filters) → MaxPool
→ Flatten
→ FC (120) → Dropout → FC (84) → FC (10)
→ Output (10 classes)
```

## 📁 Project Structure

```
cifar_cnn/
├── src/
│   ├── dataset.py      # Data loading and augmentation
│   ├── model.py        # CNN architecture definition
│   ├── train.py        # Training loop with loss plotting
│   ├── evaluate.py     # Evaluation and metrics
│   └── visualize.py    # Prediction visualization
├── main.py             # CLI entry point
├── requirements.txt    # Python dependencies
├── checkpoints/        # Saved model weights
│   └── cifar_net.pth
├── loss_curve.png      # Training loss visualization
└── predictions.png     # Sample predictions
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train the model for 10 epochs (recommended):
```bash
python main.py --mode train --epochs 10
```

Train for custom epochs with custom learning rate:
```bash
python main.py --mode train --epochs 20 --lr 0.0001
```

### Evaluation

Evaluate the trained model on the test set:
```bash
python main.py --mode evaluate
```

### Visualization

Generate sample predictions:
```bash
python src/visualize.py
```

### Web Application 🌐

**NEW!** Run the Flask web application for interactive predictions:
```bash
python app.py
```

Then open your browser to `http://localhost:5000`

Features:
- 🎨 Modern, responsive web interface
- 📤 Drag & drop image upload
- 🎯 Real-time predictions with confidence scores
- 📊 Visual probability distribution for all classes
- 📱 Mobile-friendly design

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 📈 Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Data Augmentation**: 
  - Random Crop (32x32 with padding=4)
  - Random Horizontal Flip
- **Normalization**: CIFAR-10 standard (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

## 📊 Per-Class Accuracy

| Class  | Accuracy |
|--------|----------|
| Plane  | 79.2%    |
| Car    | 90.9%    |
| Bird   | 69.9%    |
| Cat    | 62.5%    |
| Deer   | 74.3%    |
| Dog    | 62.4%    |
| Frog   | 68.9%    |
| Horse  | 72.6%    |
| Ship   | 76.9%    |
| Truck  | 88.8%    |

## 🔧 Customization

### Modify Hyperparameters

Edit `main.py` to change:
- Batch size: `--batch-size`
- Learning rate: `--lr`
- Number of epochs: `--epochs`

### Modify Architecture

Edit `src/model.py` to experiment with:
- Number of convolutional layers
- Filter sizes
- Dropout rates
- Fully connected layer dimensions

## 📝 Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy

## 🎓 Future Improvements

To achieve higher accuracy (80%+), consider:
1. **More epochs**: Train for 50-100 epochs
2. **Learning rate scheduling**: Reduce LR on plateau
3. **Deeper architecture**: Add more convolutional layers
4. **Advanced augmentation**: Cutout, MixUp, AutoAugment
5. **Regularization**: Increase dropout, add weight decay
6. **Ensemble methods**: Train multiple models and average predictions

## 📄 License

MIT License - Feel free to use this project for learning and experimentation!

## 🙏 Acknowledgments

- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch framework: [pytorch.org](https://pytorch.org/)
