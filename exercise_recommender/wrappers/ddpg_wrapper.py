from tianshou.policy.modelfree.ddpg import DDPGPolicy, TDDPGTrainingStats, DDPGTrainingStats
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

class DDPGPolicyWrapper(DDPGPolicy):

    @staticmethod
    def _mse_optimizer(
        batch: RolloutBatchProtocol,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        """
            Changes made to original:
            Gave info=batch.info parameter to critic call
        """
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act, info=batch.info).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDDPGTrainingStats:  # type: ignore
        """
            Changes made to original:
            info=batch.info is given to the self.critic
        """
        # critic
        td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        batch.weight = td  # prio-buffer
        # actor
        actor_loss = -self.critic(batch.obs, self(batch).act, info=batch.info).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()

        return DDPGTrainingStats(actor_loss=actor_loss.item(), critic_loss=critic_loss.item())  # type: ignore[return-value]
    
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
            Changes made to original:
            info = [None] * len(indices) CHANGED to info = buffer[indices].info
            info parameter is given to self.critic_old
        """
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        return self.critic_old(obs_next_batch.obs, self(obs_next_batch, model="actor_old").act, info=obs_next_batch.info)