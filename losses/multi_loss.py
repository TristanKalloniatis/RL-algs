from typing import List, Dict
from torch import Tensor, tensor


class LossManager:
    def __init__(self, loss_names: List[str], weight: float, epsilon: float):
        self._loss_names = loss_names
        self._weight = weight
        self._epsilon = epsilon
        self.loss_scales: Dict[str, float] = {loss: 0.0 for loss in loss_names}
        self.losses: Dict[str, List[float]] = {loss: [] for loss in loss_names}
        self.losses["loss"] = []

    def observe(self, losses: Dict[str, Tensor]):
        for name in losses:
            self.losses[name].append(losses[name].item())
            self.loss_scales[name] += self._weight * (
                abs(losses[name].item()) - self.loss_scales[name]
            )

    def loss(self, losses: Dict[str, Tensor]) -> Tensor:
        self.observe(losses)
        device = list(losses.values())[0].device
        total_loss = tensor(0.0, device=device)
        for name in losses:
            total_loss += losses[name] / (self.loss_scales[name] + self._epsilon)
        self.losses["loss"].append(total_loss.item())
        return total_loss
