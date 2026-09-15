import torch


def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    model.eval(); correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return correct / total

