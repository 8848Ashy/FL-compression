import torch
import torch.nn as nn


def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    model.eval(); correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return correct / total


def evaluate_model_metrics(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))
    return {"test_loss": loss_sum / max(total, 1), "test_acc": correct / max(total, 1)}
