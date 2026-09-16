import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_mnist_federated_data(root="./data", num_clients=10, images_per_client=600,
                               test_size=10000, batch_size_train=32, batch_size_test=64,
                               paired_shuffle=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    client_loaders = []
    for i in range(num_clients):
        start, end = i * images_per_client, (i + 1) * images_per_client
        # paired_shuffle attaches a dedicated generator so callers can re-seed the
        # shuffle order per (round, client) for common-random-number comparisons.
        generator = torch.Generator() if paired_shuffle else None
        client_loaders.append(DataLoader(Subset(train_dataset, list(range(start, end))), batch_size=batch_size_train,
                                         shuffle=True, generator=generator))
    test_loader = DataLoader(Subset(test_dataset, list(range(test_size))), batch_size=batch_size_test, shuffle=False)
    return client_loaders, test_loader
