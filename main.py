import argparse
import torch
import os
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True,
                        help='Mode to run: train or evaluate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader, classes = get_dataloaders(batch_size=args.batch_size)
    
    # Initialize model
    model = SimpleCNN()
    
    if args.mode == 'train':
        train_model(model, trainloader, epochs=args.epochs, lr=args.lr, device=device)
        
    elif args.mode == 'evaluate':
        model_path = './checkpoints/cifar_net.pth'
        if not os.path.exists(model_path):
            print("Model checkpoint not found. Please train first!")
            return
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate_model(model, testloader, classes, device=device)

if __name__ == '__main__':
    main()
