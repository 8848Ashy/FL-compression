"""MNIST/EMNIST loaders with reproducible IID or Dirichlet client partitions."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_mnist_federated_data(root="./data", num_clients=10, images_per_client=600,
                               test_size=10000, batch_size_train=32, batch_size_test=64,
                               paired_shuffle=False, partition="iid", dirichlet_alpha=0.5,
                               dataset="mnist", seed=42, emnist_split="balanced"):
    if num_clients < 1 or images_per_client < 1:
        raise ValueError("num_clients and images_per_client must be positive")
    if partition not in ("iid", "dirichlet"):
        raise ValueError("partition must be iid or dirichlet")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if dataset.lower() == "mnist":
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif dataset.lower() == "emnist":
        train_dataset = datasets.EMNIST(root=root, split=emnist_split, train=True,
                                        download=True, transform=transform)
        test_dataset = datasets.EMNIST(root=root, split=emnist_split, train=False,
                                       download=True, transform=transform)
    else:
        raise ValueError("dataset must be mnist or emnist")

    rng = np.random.default_rng(seed)
    needed = min(len(train_dataset), num_clients * images_per_client)
    if partition == "iid":
        indices = rng.permutation(len(train_dataset))[:needed]
        client_indices = [indices[i * images_per_client:(i + 1) * images_per_client].tolist()
                          for i in range(num_clients)]
    else:
        labels = np.asarray(train_dataset.targets)
        per_client = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            label_indices = np.flatnonzero(labels == label)
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(label_indices)).astype(int)
            for client_id, group in enumerate(np.split(label_indices, cuts)):
                per_client[client_id].extend(group.tolist())
        # Cap each shard at requested client capacity; distribute leftover examples
        # from oversized shards to clients with room, preserving class skew.
        client_indices = []
        leftovers = []
        for shard in per_client:
            rng.shuffle(shard)
            client_indices.append(shard[:images_per_client])
            leftovers.extend(shard[images_per_client:])
        rng.shuffle(leftovers)
        for shard in client_indices:
            room = images_per_client - len(shard)
            if room:
                shard.extend(leftovers[:room])
                del leftovers[:room]

    client_loaders = []
    for indices in client_indices:
        generator = torch.Generator().manual_seed(seed) if paired_shuffle else None
        client_loaders.append(DataLoader(
            Subset(train_dataset, indices), batch_size=batch_size_train,
            shuffle=bool(indices), generator=generator,
        ))
    test_size = min(test_size, len(test_dataset))
    test_loader = DataLoader(Subset(test_dataset, list(range(test_size))),
                             batch_size=batch_size_test, shuffle=False)
    return client_loaders, test_loader
