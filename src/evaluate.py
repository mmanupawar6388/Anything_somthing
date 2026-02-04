import torch
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """Helper function to un-normalize and display an image"""
    img = img / 2 + 0.5  # unnormalize (assuming mean=0.5, std=0.5 approx for visual simplicity, 
                         # though we used specific mean/std in dataset.py, this is a quick approx)
    # The actual normalization was: mean=(0.4914, ...), std=(0.2023, ...)
    # To be precise we should reverse that, but for quick visual inspection typical un-norm is fine.
    # Let's do a more proper un-norm based on the values in dataset.py to avoid confusion.
    # mean = np.array([0.4914, 0.4822, 0.4465])
    # std = np.array([0.2023, 0.1994, 0.2010])
    # img = img * std + mean
    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def evaluate_model(model, testloader, classes, device='cpu'):
    """
    Evaluates the model on the test set and prints accuracy.
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # No gradient needed for evaluation
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
    
    # Class-wise accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s} : {100 * class_correct[i] / class_total[i]:2f} %')
