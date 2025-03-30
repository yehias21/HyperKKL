from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
from omegaconf import DictConfig


def activation_fn(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Activation function {activation} not supported')


def get_model(mapper_config: DictConfig):
    match mapper_config.type.lower():
        case "mlp":
            try:
                return MLP(in_dim=mapper_config.input_size, out_dim=mapper_config.output_size,
                           hidden_dim=mapper_config.get('hidden_dim', None),
                           activation=mapper_config.get('activation', None),
                           weight_path=mapper_config.get('weight_path', None),
                           technique=mapper_config.update_method)
            except Exception as e:
                raise ValueError(f"Error in initializing MLP model: {e}")

        case _:
            raise ValueError(f"Unknown mapper type: {mapper_config.type}")


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim: Optional[list],
                 activation: list[str], weight_path: str = None, technique: str = 'backprop'):
        super(MLP, self).__init__()
        self.activation = activation
        self.technique = technique.lower()
        self.mlp = nn.ModuleList()
        self._gen_mlp(in_dim, out_dim, hidden_dim, activation)
        # Fine_tune or use pretrained weights as initial weights
        if weight_path is not None:
            try:
                self.load_state_dict(torch.load(weight_path), strict=True)
            except FileNotFoundError:
                print(f"File {weight_path} not found")
        if technique == "delta":
            # assert weight_path is not None, "Weight path is required for delta technique"
            for layer in self.mlp:
                for param in layer.parameters():
                    param.requires_grad = False

    def _gen_mlp(self, in_dim, out_dim, hidden_dim, activation):
        if hidden_dim is None:
            self.mlp.append(nn.Linear(in_dim, out_dim))
        else:
            assert len(hidden_dim) == len(activation)
            self.mlp.append(nn.Linear(in_dim, hidden_dim[0]))
            self.mlp.append(activation_fn(activation[0]))

            for i in range(1, len(hidden_dim)):
                self.mlp.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                self.mlp.append(activation_fn(activation[i]))
            self.mlp.append(nn.Linear(hidden_dim[-1], out_dim))

    def _forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

    def _forward_delta(self, x, delta_weights: dict[str, torch.Tensor]):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # check dict if weight is present in one line
                w_final = module.weight.detach().clone() + delta_weights.get(f'{name}.weight', torch.Tensor([0]))
                bias = module.bias.detach().clone() + delta_weights.get(f'{name}.bias', torch.Tensor([0]))
                temp = []
                # fixme: this is a temporary fix, need to find a better way... and bias is assumed to be 2D and not 3D
                for i in range(w_final.shape[0]):
                    temp.append(F.linear(x[i], w_final[i], bias))
                x = torch.stack(temp)
                x = activation_fn(module.__class__.__name__.lower())(x)
        return x

    def _forward_hnn(self, x, weights: dict[str, torch.Tensor]):
        for idx, (name, weight) in enumerate(weights.items()):
            temp = []
            for i in range(weight.shape[0]):
                temp.append(F.linear(x[i], weight[i]))
            x = torch.stack(temp)
            x = activation_fn(self.activation[idx])(x)
        return x

    def forward(self, x, weight: dict[str, torch.Tensor] = None):
        if weight:
            if self.technique == 'delta':
                return self._forward_delta(x, weight)
            else:
                return self._forward_hnn(x, weight)
        else:
            assert self.technique in ['backprop', 'none'], "Provide the weights of Hypernetwork"
            return self._forward(x)
