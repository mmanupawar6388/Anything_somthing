import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=64, num_workers=2):
    """
    Creates and returns the CIFAR-10 training and test data loaders.
    Includes data augmentation for training.
    """
    
    # Transformations
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    # Datasets
    # We will download to a 'data' folder
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=test_transform)
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                               shuffle=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size*2, 
                                              shuffle=False, num_workers=num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

if __name__ == "__main__":
    train_dl, test_dl, classes = get_data_loaders()
    print(f"Train batches: {len(train_dl)}")
    print(f"Test batches: {len(test_dl)}")
    print(f"Classes: {classes}")
