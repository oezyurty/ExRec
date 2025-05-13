from tianshou.policy.modelfree.td3 import TD3Policy, TD3TrainingStats, TTD3TrainingStats

from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)

import torch
from typing import Any, Generic, Literal, Self, TypeVar, cast
from tianshou.data import Batch, ReplayBuffer
import numpy as np

class TD3PolicyWrapper(TD3Policy):

    @staticmethod
    def _mse_optimizer(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        """
            Changes made to original:
            info=batch.info parameter is given to critic call
        """
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act, info=batch.info).flatten()
        target_q = batch.returns.flatten()
        # TODO: Remove after debugging
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss
    
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
            Changes made to original:
            info = [None] * len(indices) CHANGED to info = buffer[indices].info
            info parameter is given to self.critic_old and self.critic2_old
        """
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        act_ = self(obs_next_batch, model="actor_old").act
        noise = torch.randn(size=act_.shape, device=act_.device) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        act_ += noise
        return torch.min(
            self.critic_old(obs_next_batch.obs, act_, info=obs_next_batch.info),
            self.critic2_old(obs_next_batch.obs, act_, info=obs_next_batch.info),
        )
    
    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TTD3TrainingStats:  # type: ignore
        """
            Changes made to the original
            info=batch.info is given to the self.critic
        """
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self.update_actor_freq == 0:
            actor_loss = -self.critic(batch.obs, self(batch, eps=0.0).act, info=batch.info).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1

        return TD3TrainingStats(  # type: ignore[return-value]
            actor_loss=self._last,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )


