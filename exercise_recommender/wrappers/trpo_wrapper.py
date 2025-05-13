from tianshou.policy.modelfree.trpo import TRPOPolicy, TRPOTrainingStats, TTRPOTrainingStats
import warnings
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
import torch
from torch.distributions import kl_divergence
import gymnasium as gym
import torch.nn.functional as F
from typing import Any, Generic, Literal, Self, TypeVar, cast
from tianshou.data import Batch, ReplayBuffer
import numpy as np

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.policy import NPGPolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.npg import NPGTrainingStats
from tianshou.policy.modelfree.pg import TDistFnDiscrOrCont
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as DiscreteActor
from tianshou.utils.net.discrete import Critic as DiscreteCritic
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol

class TRPOPolicyWrapper(TRPOPolicy):
    def _compute_returns(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        """
            Changes made to the original
            info=minibatch.info is given to self.critic calls
        """
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                v_s.append(self.critic(minibatch.obs, info=minibatch.info))
                v_s_.append(self.critic(minibatch.obs_next, info=minibatch.info))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Empirical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        # TODO: see todo in PGPolicy.process_fn
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    def learn(  # type: ignore
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
        **kwargs: Any,
    ) -> TTRPOTrainingStats:
        """
            Changes made to the original code:
            info=minibatch.info is given to self.critic
        """
        actor_losses, vf_losses, step_sizes, kls = [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self(minibatch).dist  # TODO could come from batch
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                actor_loss = -(ratio * minibatch.adv).mean()
                flat_grads = self._get_flat_grad(actor_loss, self.actor, retain_graph=True).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, nsteps=10)

                # stepsize: calculate max stepsize constrained by kl bound
                step_size = torch.sqrt(
                    2
                    * self.max_kl
                    / (search_direction * self._MVP(search_direction, flat_kl_grad)).sum(
                        0,
                        keepdim=True,
                    ),
                )

                # stepsize: linesearch stepsize
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.actor.parameters()],
                    )
                    for i in range(self.max_backtracks):
                        new_flat_params = flat_params + step_size * search_direction
                        self._set_from_flat_params(self.actor, new_flat_params)
                        # calculate kl and if in bound, loss actually down
                        new_dist = self(minibatch).dist
                        new_dratio = (
                            (new_dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                        )
                        new_dratio = new_dratio.reshape(new_dratio.size(0), -1).transpose(0, 1)
                        new_actor_loss = -(new_dratio * minibatch.adv).mean()
                        kl = kl_divergence(old_dist, new_dist).mean()

                        if kl < self.max_kl and new_actor_loss < actor_loss:
                            if i > 0:
                                warnings.warn(f"Backtracking to step {i}.")
                            break
                        if i < self.max_backtracks - 1:
                            step_size = step_size * self.backtrack_coeff
                        else:
                            self._set_from_flat_params(self.actor, new_flat_params)
                            step_size = torch.tensor([0.0])
                            warnings.warn(
                                "Line search failed! It seems hyperparamters"
                                " are poor and need to be changed.",
                            )

                # optimize critic
                # TODO: remove type-ignore once the top-level type-ignore is removed
                for _ in range(self.optim_critic_iters):  # type: ignore
                    value = self.critic(minibatch.obs, info=minibatch.info).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
                kls.append(kl.item())

        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        kl_summary_stat = SequenceSummaryStats.from_sequence(kls)
        step_size_stat = SequenceSummaryStats.from_sequence(step_sizes)

        return TRPOTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            kl=kl_summary_stat,
            step_size=step_size_stat,
        )