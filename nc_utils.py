import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import tqdm
from typing import Dict, Hashable, Optional


_NUM_CLASSES = 100

def get_one_hot(targets, num_classes: Optional[int] = None):
    if num_classes is None:
        num_classes = _NUM_CLASSES
    if torch.max(targets) > 1 or targets.ndim < 2:
        return F.one_hot(targets, num_classes=num_classes)
    else:
        return targets


def nearest_class_classifier_accuracy(model: torch.nn.Module, class_means: Dict[Hashable, torch.Tensor],
                                      data_loader: DataLoader, pbar_desc: str = 'NCC: ') -> Dict[Hashable, float]:
    """Check the accuracy of the 'Nearest class-mean classifier' for the different embeddings.

    :param model: Model to evaluate on
    :param class_means: Class means in the embeddings. Dict mapping layer name to tensor of means
    :param data_loader: Data to evaluate
    :param pbar_desc: Description to use in the progress bar
    :returns: Dict of embedding layer name mapping to that layer's NCC accuracy

    """
    # Return immediately if there are no embeddings to compute
    if len(class_means) == 0:
        return {}

    model.eval()
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
            labels = one_hot_targets.argmax(dim=1)
            batch_correct = preds.eq(labels).sum().item()
            correct[layer_name] += batch_correct
            pbar_str += f'{layer_name}: {batch_correct / len(one_hot_targets):<5.3G}, '

        pbar.set_description(pbar_desc + pbar_str)
    pbar.close()

    accuracy = {layer_name: layer_correct / total_samples
                for layer_name, layer_correct in correct.items()}

    return accuracy


def embedding_classifier_accuracy(model: torch.nn.Module, class_means: Dict[Hashable, torch.Tensor],
                                  train_data_loader: DataLoader, test_data_loader: DataLoader,
                                  pbar_desc: str = 'LinClass: ', train_epochs: int = 2, init_class_means: bool = False,
                                  lr=10
                                  ):
    if len(class_means) == 0:
        return {}
    if init_class_means:  # TODO(marius): Make linear classifiers be initialized with class means
        raise NotImplementedError("Init with class means not implemented")

    loss_fcn = torch.nn.MSELoss()
    num_classes = next(iter(class_means.values())).size()[0]

    model.eval()
    device = next(model.parameters()).device

    inputs = next(iter(train_data_loader))[0].to(device)
    embeddings_dict = model(inputs)[1]

    # Initialize linear classifiers for each embedding
    linear_classifiers = {}
    params = []
    for layer_name, embedding_activation in embeddings_dict.items():
        embedding_dim = embedding_activation.size()[1:].numel()
        linear_proj = torch.nn.Linear(
            in_features=embedding_dim, out_features=num_classes, bias=True  # TODO(marius): Should this be False?
        )
        linear_proj.to(device)
        linear_classifiers[layer_name] = linear_proj
        params += list(linear_classifiers[layer_name].parameters())

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    # Train linear classifiers
    for epoch in range(train_epochs):
        pbar_batch = tqdm.tqdm(train_data_loader, position=1, leave=False, ncols=None)
        pbar_batch.set_description(pbar_desc + 'E[{}/{}]'.format(epoch+1, train_epochs))

        total_samples = 0
        loss: torch.Tensor = 0.0
        for inputs, targets in pbar_batch:
            inputs, targets = inputs.to(device), targets.to(device)
            one_hot_targets = get_one_hot(targets, num_classes=num_classes)
            total_samples += len(one_hot_targets)

            optimizer.zero_grad()
            with torch.no_grad():
                outputs, embeddings = model(inputs)

            for layer_name, embedding_activation in embeddings.items():
                flattened_embedding = embedding_activation.view(embedding_activation.size(0), -1)
                class_pred = linear_classifiers[layer_name](flattened_embedding)
                loss += loss_fcn(class_pred, one_hot_targets)

            pbar_batch.set_description(pbar_desc + 'E[{}/{}]'.format(epoch + 1, train_epochs)
                                       + ' Loss: {:5.3G}'.format(loss.item() / len(one_hot_targets)))
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

    train_accuracy = linear_classifier_accuracy(model, linear_classifiers, num_classes, data_loader=train_data_loader)
    test_accuracy = linear_classifier_accuracy(model, linear_classifiers, num_classes, data_loader=test_data_loader)

    return train_accuracy, test_accuracy


def linear_classifier_accuracy(model: torch.nn.Module, linear_classifiers: Dict[Hashable, torch.nn.Module], num_classes: int,
                               data_loader: DataLoader, pbar_desc: str = 'LinClass eval: ') -> Dict[Hashable, float]:
    """Check the accuracy of the linear classifiers for each of the different embeddings.

    :param model: Model to evaluate on
    :param linear_classifiers: Linear classifiers for each of the embeddings. Dict mapping embedding name to classifier
    :param num_classes: Number of classes that the classifiers output to.
    :param data_loader: Data to evaluate
    :param pbar_desc: Description to use in the progress bar
    :returns: Dict of embedding layer name mapping to that layer's NCC accuracy

    """
    if len(linear_classifiers) == 0:
        return {}

    model.eval()
    device = next(model.parameters()).device

    # Evaluate linear classifier over all batches
    correct = {layer_name: .0 for layer_name in linear_classifiers.keys()}
    pbar_batch = tqdm.tqdm(data_loader, position=1, leave=False, ncols=None)
    pbar_batch.set_description(pbar_desc + 'Evaluation')

    total_samples = 0
    for inputs, targets in pbar_batch:
        pbar_str = ''
        inputs, targets = inputs.to(device), targets.to(device)
        one_hot_targets = get_one_hot(targets, num_classes=num_classes)
        labels = one_hot_targets.argmax(dim=1)
        total_samples += len(one_hot_targets)

        with torch.no_grad():
            outputs, embeddings = model(inputs)

        # Calculate number of correct classifications for each embedding
        for layer_name, embedding_activation in embeddings.items():
            embedding_flattened = embedding_activation.view(embedding_activation.size(0), -1)

            classifier_out = linear_classifiers[layer_name](embedding_flattened)
            class_preds = classifier_out.argmax(dim=1)
            batch_correct = class_preds.eq(labels).sum().item()
            correct[layer_name] += batch_correct

            pbar_str += f'{layer_name}: {batch_correct / len(labels):<5.3G}, '
        pbar_batch.set_description(pbar_desc + pbar_str)

    accuracy = {layer_name: layer_correct / total_samples
                for layer_name, layer_correct in correct.items()}

    return accuracy


if __name__ == '__main__':
    pass
