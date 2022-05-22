import torch
import torch.functional as F
import torch.nn as nn
import torchvision.models as models

from typing import Dict, Optional, Hashable
from collections import OrderedDict


class HookedOutput(torch.Tensor):
    """Output of a ForwardHookedModel with "normal" model outputs as well as a self.hook_outputs."""
    hook_outputs: Dict[Hashable, torch.Tensor]

    def __init__(self, out: torch.Tensor, hook_outputs: Dict[Hashable, torch.Tensor]):
        super().__init__(out)
        self.hook_outputs = hook_outputs


class ForwardHookedModel(nn.Module):
    """Wrapper around a model with intermediate layer forward hooks"""
    def __init__(self, base_model: nn.Module, output_layers: Optional[Dict[Hashable, float]] = None,
                 prediction_loss_weight: float = 1, *args):
        """Wrap a base model with forward hooks for output.

        Forward call returns a HookedOuput.

        :param base_model: Base model to wrap
        :param output_layers: Layers for which to create forward hooks. Dict of identifier: loss_weight. Default: None
            (i.e. {'layer_name': 0.01, 'other_layer_name': 0.2})
        :param prediction_loss_weight: Weighting of ordinary output loss. Default: 1
        :param args: Args to pass to nn.Module, if any
        """
        # Init and store base model
        super().__init__(*args)
        self.base_model = base_model

        # Parameters
        self.prediction_loss_weight: float = prediction_loss_weight
        self.output_layers_weights_dict: Dict[Hashable, float] = output_layers if output_layers is not None else {}
        self.hook_outputs: Dict[Hashable, torch.Tensor] = OrderedDict()
        self.fwd_hooks = []

        # Register hooks
        for i, layer_name in enumerate(list(self.base_model._modules.keys())):  # TODO(marius): Iterate over internal layers too, not just top level
            if i in self.output_layers_weights_dict.keys() or layer_name in self.output_layers_weights_dict.keys():
                self.fwd_hooks.append(
                    getattr(self.base_model, layer_name).register_forward_hook(self.forward_hook(layer_name))
                )

    def forward_hook(self, layer_id):
        def hook(module, inputs, outputs):
            self.hook_outputs[layer_id] = outputs
        return hook

    def forward(self, x) -> HookedOutput:
        out = self.base_model(x)
        return HookedOutput(out, self.hook_outputs)


def _test():
    def get_model(model_cfg: Dict):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_model = models.resnet18(pretrained=False)
        # base_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        base_model.to(device)

        # out_layers = {f'layer{i}': 1 for i in range(3, 5)}
        prediction_loss_weight = model_cfg['prediction_loss_weight']
        out_layers = model_cfg['embedding_layers']
        ret_model = ForwardHookedModel(base_model, out_layers, prediction_loss_weight=prediction_loss_weight).to(device)
        return ret_model

    _ret_model = get_model(dict(
        prediction_loss_weight=0.001,
        embedding_layers={'layer3': 1, 'layer4': 1},
    ))
    print(_ret_model)


if __name__ == "__main__":
    _test()
