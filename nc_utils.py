import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import tqdm
from typing import Dict, Union, List, Callable, Tuple


_NUM_CLASSES = 100

def get_one_hot(targets):
    if torch.max(targets) > 1 or targets.ndim < 2:
        return F.one_hot(targets, num_classes=_NUM_CLASSES)
    else:
        return targets


def nearest_class_classifier_accuracy(model: torch.nn.Module, class_means: Dict[Union[str, int], torch.Tensor], data_loader: DataLoader, pbar_desc='NCC: ') -> Dict[Union[str, int], float]:
    # Return immediately if there are no embeddings to compute
    if len(class_means) == 0:
        return {}

    device = next(model.parameters()).device  # Get model device

    correct = {layer_name: .0 for layer_name in class_means.keys()}
    flat_class_means = {name: means.reshape(means.shape[0], -1) for name, means in class_means.items()}

    pbar = tqdm.tqdm(data_loader, position=1, leave=False, ncols=None)
    pbar.set_description(pbar_desc)
    total_samples = 0
    for (inputs, targets) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        one_hot_targets = get_one_hot(targets)
        total_samples += len(one_hot_targets)
        with torch.no_grad():
            outputs, embeddings = model(inputs)

        pbar_str = ''
        for layer_name, embedding_activation in embeddings.items():
            flat_activation = embedding_activation.view(embedding_activation.shape[0], -1)
            means = flat_class_means[layer_name]
            dists = torch.cdist(flat_activation.unsqueeze(0), means.unsqueeze(0)).squeeze(0)
            preds = dists.argmin(dim=1)
            labels = one_hot_targets.argmax(dim=1)  # TODO(marius): fix one-hot encoding
            batch_correct = preds.eq(labels).sum().item()
            correct[layer_name] += batch_correct
            pbar_str += f'{layer_name}: {batch_correct / len(one_hot_targets):<5.3G}, '

        pbar.set_description(pbar_desc + pbar_str)
    pbar.close()

    for layer_name, total_correct in correct.items():
        correct[layer_name] = total_correct / total_samples

    return correct


if __name__ == '__main__':
    pass
