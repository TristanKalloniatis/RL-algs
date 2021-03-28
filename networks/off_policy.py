from copy import deepcopy
from typing import List
from torch import Tensor
from torch.nn import Module


class OffPolicyNetworkFactory:
    def __init__(self, network: Module, polyak_weight: float):
        self.online_network = deepcopy(network)
        self.target_network = deepcopy(network)
        self.polyak_weight = polyak_weight


    def __call__(self, inputs: List[Tensor], use_online: bool = True) -> Tensor:
        return self.online_network(inputs) if use_online else self.target_network(inputs).detach()

    def synchronise(self, use_polyak: bool = True):
        weight = self.polyak_weight if use_polyak else 1.0
        for online, target in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target.data.copy_(weight * online.data + (1.0 - weight) * target.data)
