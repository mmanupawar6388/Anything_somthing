import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SimpleCNN
from src.dataset import get_dataloaders

def imshow(img, title=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(12, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Saved predictions.png")

def visualize_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    # Use batch_size=4 for visualization
    _, testloader, classes = get_dataloaders(batch_size=4)
    
    # Load model
    model = SimpleCNN()
    model_path = './checkpoints/cifar_net.pth'
    if not os.path.exists(model_path):
        print("Model checkpoint not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get a batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images.to(device)
    
    # Predict
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Prepare labels for display
    ground_truth = [classes[labels[j]] for j in range(4)]
    predictions = [classes[predicted[j]] for j in range(4)]
    
    title = f"Ground Truth: {ground_truth}\nPredicted:    {predictions}"
    print(title)
            
    # Show
    imshow(torchvision.utils.make_grid(images.cpu()), title)

if __name__ == '__main__':
    visualize_predictions()
