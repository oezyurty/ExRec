from tianshou.policy.modelfree.discrete_sac import DiscreteSACPolicy, DiscreteSACTrainingStats, TDiscreteSACTrainingStats

from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from overrides import override
from torch.distributions import Categorical

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import SACPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.sac import SACTrainingStats
from tianshou.utils.net.discrete import Actor, Critic


class DiscreteSACPolicyWrapper(DiscreteSACPolicy):

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=buffer[indices].info,
        )  # obs_next: s_{t+n}
        obs_next_result = self(obs_next_batch)
        dist = obs_next_result.dist
        min_val = torch.min(
            self.critic_old(obs_next_batch.obs, info=obs_next_batch.info),
            self.critic2_old(obs_next_batch.obs, info=obs_next_batch.info),
        )
        target_q = dist.probs * min_val
        return target_q.sum(dim=-1) + self.alpha * dist.entropy()

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDiscreteSACTrainingStats:  # type: ignore
        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = to_torch(batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long)

        # critic 1
        current_q1 = self.critic(batch.obs, info=batch.info).gather(1, act).flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()

        self.critic_optim.zero_grad()
        critic1_loss.backward()
        self.critic_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs, info=batch.info).gather(1, act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic(batch.obs, info=batch.info)
            current_q2a = self.critic2(batch.obs, info=batch.info)
            q = torch.min(current_q1a, current_q2a)
        actor_loss = -(self.alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.is_auto_alpha:
            log_prob = -entropy.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        if self.is_auto_alpha:
            self.alpha = cast(torch.Tensor, self.alpha)

        return DiscreteSACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            alpha_loss=None if not self.is_auto_alpha else alpha_loss.item(),
        )

    _TArrOrActBatch = TypeVar("_TArrOrActBatch", bound="np.ndarray | ActBatchProtocol")