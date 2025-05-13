from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import DQNPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.dqn import DQNTrainingStats
from tianshou.policy.modelfree.c51 import C51Policy, C51TrainingStats, TC51TrainingStats

device = "cuda" if torch.cuda.is_available() else "cpu"

class C51PolicyWrapper(C51Policy):
    
    def _target_dist(self, batch: RolloutBatchProtocol) -> torch.Tensor:
        obs_next_batch = Batch(obs=batch.obs_next, info=[None] * len(batch))
        if self._target:
            act = self(obs_next_batch).act
            next_dist = self(obs_next_batch, model="model_old").logits
        else:
            next_batch = self(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        target_support = batch.returns.clamp(self._v_min, self._v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_support = target_support.to(device)
        support = self.support.view(1,-1,1).to(device)
        target_dist = (
            1 - (target_support.unsqueeze(1) - support).abs() / self.delta_z
        ).clamp(0, 1) * next_dist.unsqueeze(1)
        return target_dist.sum(-1)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TC51TrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        weight = batch.pop("weight", 1.0)
        curr_dist = self(batch).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy * weight).mean()
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/agent.py L94-100
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return C51TrainingStats(loss=loss.item())  # type: ignore[return-value]

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        support = self.support.to(device)
        support = support.view(1,1,-1)
        x = (logits * support).sum(2)
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = x.min() - x.max() - 1.0
            x = x + to_torch_as(1 - mask, x) * min_value
        return x