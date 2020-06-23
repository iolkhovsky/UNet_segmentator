from transform_utils import tensor_bcyx2byxc
from torch.nn import CrossEntropyLoss
import torch
import numpy as np


def compute_loss(prediction_tensor, target_maps, weights, classes_cnt):
    pred = tensor_bcyx2byxc(prediction_tensor)
    pred = torch.reshape(pred, [-1, classes_cnt])

    average_weights = None
    if type(weights) == np.ndarray:
        average_weights = np.mean(weights, axis=0)
    elif type(weights) == torch.Tensor:
        average_weights = torch.mean(weights, dim=0)
    else:
        raise RuntimeError("Unexpected weights type"+str(type(weights)))

    criterion = torch.nn.BCEWithLogitsLoss()
    if classes_cnt > 2:
        criterion = torch.nn.CrossEntropyLoss()
        target = np.argmax(target_maps, axis=1)
        target = target.flatten()
    else:
        target = tensor_bcyx2byxc(target_maps).float()
        target = torch.reshape(target, [-1, classes_cnt])

    loss = criterion(pred, target)
    return loss
