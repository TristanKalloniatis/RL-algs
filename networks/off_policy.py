from copy import deepcopy


class OffPolicyNetworkFactory:
    def __init__(self, network, polyak_weight: float):
        self.online_network = deepcopy(network)
        self.target_network = deepcopy(network)
        self.polyak_weight = polyak_weight

    def synchronise(self, use_polyak: bool = True):
        weight = self.polyak_weight if use_polyak else 1.0
        for online, target in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            target.data.copy_(weight * online.data + (1.0 - weight) * target.data)
