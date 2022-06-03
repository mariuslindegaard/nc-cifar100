import torch
import torch.nn as nn
import torch.nn.functional as F

from ForwardHookedModel import ForwardHookedModel

import nc_utils

import dataclasses
from typing import Optional, Dict, Tuple, List, Callable, Hashable
import warnings


@dataclasses.dataclass
class MultipleCriterions:
    prediction_criterion: Optional[nn.modules.Module]
    hook_criterions: Dict[Hashable, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]  # <- Criterions for hooks, must be of same length as total number of fwd hooks
    prediction_loss_weight: Optional[float]
    hook_loss_weights_dict: Dict[Hashable, float]
    last_losses: Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, Dict[Hashable, torch.Tensor]]]] = None

    def __call__(self, output, targets) -> torch.Tensor:
        # Unpack input to predictions and intermediate embeddings
        predictions, embedding_dict = output
        assert set(embedding_dict.keys()).issubset(set(self.hook_criterions.keys()))

        # Calculate output loss
        pred_loss = self.prediction_criterion(predictions, targets) if self.prediction_criterion is not None else 0

        # Calculate embedding/hook losses
        hook_losses = {embedding_str: None for embedding_str in embedding_dict.keys()}
        for embedding_str, embedding_value in embedding_dict.items():
            criterion = self.hook_criterions[embedding_str]
            hook_losses[embedding_str] = criterion(embedding_value, targets)

        # Sum together weighted losses
        total_loss = self.prediction_loss_weight * pred_loss
        for embedding_str in embedding_dict.keys():
            total_loss += hook_losses[embedding_str] * self.hook_loss_weights_dict[embedding_str]

        self.last_losses = total_loss, (pred_loss, hook_losses)

        return total_loss


class Criterions:
    _epsilon = 1E-12  # TODO(marius): Make epsilon dynamic
    _min_required_samples_for_valid_loss = 4  # Should be at least 2

    @classmethod
    def set_epsilon(cls, epsilon):
        if cls._epsilon is not None:
            warnings.warn("Epsilon already set in Criterions")
            if cls._epsilon != epsilon:
                warnings.warn(f"Epsilon changed from {cls._epsilon} to {epsilon}")

        cls._epsilon = epsilon


    @staticmethod
    def get_CDNV_criterion(cdnv_weighting: Dict[Hashable, float], prediction_loss: nn.modules.Module, prediction_weighting: float) -> MultipleCriterions:

        pred_criterion = prediction_loss
        pred_weight = prediction_weighting
        hook_criterions = {layer_name: Criterions.CDNV_loss for layer_name in cdnv_weighting.keys()}
        hook_loss_weights = cdnv_weighting
        crits = MultipleCriterions(pred_criterion, hook_criterions, pred_weight, hook_loss_weights)

        return crits

    @classmethod
    def CDNV_loss(cls, embedding: torch.Tensor, targets: torch.Tensor, _epsilon=None):
        if _epsilon is None:
            assert cls._epsilon is not None, "Epsilon not set in CDNV_loss"
            _epsilon = cls._epsilon

        # Make targets one-hot if they are not already
        one_hot_targets = nc_utils.get_one_hot(targets)
        num_classes = one_hot_targets.shape[-1]

        # Calculate loss for each embedding
        # for weighting, (layer, activations) in zip(_cdnv_weighting, embeddings.items()):  # TODO(marius): Paralellizable, but probably little to gain...
        class_frequency = torch.zeros(num_classes)
        class_mean = torch.zeros((num_classes, embedding[0].nelement()))
        class_var = torch.zeros(num_classes)
        # Save a mask of valid values, making them invalid if they do not appear often enough
        valid_values = torch.ones((num_classes, num_classes), dtype=torch.bool)

        for cls_idx in range(num_classes):  # TODO(marius): Vectorizable, I think...
            # Activations: (mxc)
            # Get incdices of all datapoints belonging to that class
            idxs = (torch.argmax(one_hot_targets, dim=-1) == cls_idx).nonzero(as_tuple=True)[0]  # Todo(marius): Move to outside outer loop
            class_frequency[cls_idx] = len(idxs)

            # If we have too few datapoints remove that class from valid values mask
            if len(idxs) < cls._min_required_samples_for_valid_loss:
                # class_mean[cls_idx] = torch.inf
                # class_var[cls_idx] *= torch.nan
                valid_values[cls_idx, :] = 0
                valid_values[:, cls_idx] = 0
                continue

            # Calculate in-class mean and variance
            var, mean = torch.var_mean(embedding[idxs, :], dim=0)  # Todo(marius): Could use Bessels correction for variance
            class_mean[cls_idx] = torch.flatten(mean)
            class_var[cls_idx] = torch.sum(var)

        # Weight by class frequency, linear in each classs freq.
        class_frequency = torch.nn.functional.normalize(class_frequency, dim=0, p=1)
        class_var = class_var * class_frequency

        # Get matrices of class var sums and class mean differences
        var_sums = class_var.reshape(-1, 1) + class_var
        mean_diffs = torch.cdist(class_mean, class_mean) ** 2

        # Calculate "class-distance normalized variance"
        cdnv = var_sums / torch.clamp(2*mean_diffs, min=_epsilon)  # Epsilon to avoid div by 0  # TODO(marius): Try using max(2*mean_diffs, _epsilon) instead

        # Fill diagonal with 0s since we should not have any contribution from within class "cdnv"
        cdnv.fill_diagonal_(0)

        # Check and correct for nans from classes with too few samples:
        cdnv = cdnv * valid_values

        # if torch.any(torch.isnan(cdnv)):
        #     warnings.warn("Batch with one or zero class samples found")
        # cdnv = torch.nan_to_num(cdnv, nan=0.0)

        # Add to loss
        loss = torch.sum(cdnv)

        if torch.isnan(loss):
            raise Exception("Loss should not be nan!")

        return loss
