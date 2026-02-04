import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

def train_model(model, trainloader, epochs=10, lr=0.001, device='cpu'):
    """
    Trains the model and saves the best checkpoint.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {device}...")
    model.to(device)
    
    loss_history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                avg_loss = running_loss / 100
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                loss_history.append(avg_loss)
                running_loss = 0.0
                
    print('Finished Training')
    
    # Plot loss
    plt.figure()
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    print("Saved loss_curve.png")
    
    # Save the model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = './checkpoints/cifar_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model
