from tianshou.policy.modelfree.sac import SACPolicy, SACTrainingStats, TSACTrainingStats
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
    DistLogProbBatchProtocol
)
import torch
from typing import Any, Generic, Literal, Self, TypeVar, cast
from tianshou.data import Batch, ReplayBuffer
import numpy as np

from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.conversion import to_optional_float
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.optim import clone_optimizer

from torch.distributions import Independent, Normal

class SACPolicyWrapper(SACPolicy):

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

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
            Changes made to the original code:
            info=obs_next_batch.info is given to self.critic_old and self.critic2_old
        """
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        obs_next_result = self(obs_next_batch)
        act_ = obs_next_result.act
        min_val = torch.min(
                self.critic_old(obs_next_batch.obs, act_, info=obs_next_batch.info),
                self.critic2_old(obs_next_batch.obs, act_, info=obs_next_batch.info),
            )
        min_val = min_val.unsqueeze(1)
        diff = self.alpha * obs_next_result.log_prob
        res = min_val-diff
        return res

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TSACTrainingStats:  # type: ignore
        """
            Changes made to the original code:
            info=batch.info is goven to self.critic and self.critic2 for 
                current_q1a and current_q2a calculation
        """
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic(batch.obs, act, info=batch.info).flatten()
        current_q2a = self.critic2(batch.obs, act, info=batch.info).flatten()
        actor_loss = (
            self.alpha * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        alpha_loss = None

        if self.is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self.target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        return SACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=to_optional_float(self.alpha),
            alpha_loss=to_optional_float(alpha_loss),
        )
    
    # TODO: violates Liskov substitution principle
    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> DistLogProbBatchProtocol:
        (loc_B, scale_B), hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
        dist = Independent(Normal(loc=loc_B, scale=scale_B), 1)
        if self.deterministic_eval and not self.is_within_training_step:
            act_B = dist.mode
        else:
            act_B = dist.rsample()
        log_prob = dist.log_prob(act_B).unsqueeze(-1)

        #squashed_action = torch.tanh(act_B)
        #log_prob = correct_log_prob_gaussian_tanh(log_prob, squashed_action)
        norms = act_B.norm(p=2, dim=1, keepdim=True)
        act_B = act_B/norms
        act_B = act_B * np.sqrt(act_B.shape[1])
        result = Batch(
            logits=(loc_B, scale_B),
            act=act_B,
            state=hidden_BH,
            dist=dist,
            log_prob=log_prob,
        )
        return cast(DistLogProbBatchProtocol, result)

