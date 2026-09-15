import copy
import torch.optim as optim
import torch.nn as nn
from utils.state_dict import state_dict_subtract


def local_train(global_model, dataloader, epochs=2, lr=0.05):
    device = next(global_model.parameters()).device
    local_model = copy.deepcopy(global_model); optimizer = optim.SGD(local_model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    local_model.train()
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); loss = criterion(local_model(images), labels); loss.backward(); optimizer.step()
    return local_model.state_dict()


def local_train_delta(global_model, dataloader, epochs=2, lr=0.05):
    global_state = copy.deepcopy(global_model.state_dict()); local_state = local_train(global_model, dataloader, epochs, lr)
    return state_dict_subtract(local_state, global_state)

